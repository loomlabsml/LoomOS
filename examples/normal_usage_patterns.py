"""
LoomOS RL Gym - Normal Module Usage Patterns

This file demonstrates all the ways you can use LoomOS RL Gym
as a normal Python module in your projects.
"""

# ============================================================================
# NORMAL IMPORTS (like any Python package)
# ============================================================================

# Import specific components
from rl.loom_gym import LoomRLGym, MathEnvironment, GameEnvironment
from rl.gym_integration import create_integrated_rl_system
from rl.ppo_trainer import PPOConfig, create_ppo_for_gym

# Import entire modules
import rl.loom_gym as gym
import rl.gym_integration as integration

# Standard library imports
import asyncio
import numpy as np

# ============================================================================
# NORMAL CLASS USAGE
# ============================================================================

class MyCustomAgent:
    """Normal class that uses LoomOS RL Gym"""
    
    def __init__(self):
        # Normal object creation
        self.gym = LoomRLGym()
        self.environments = {}
        
    def setup_environments(self):
        """Normal method that sets up gym environments"""
        # Create environments normally
        math_env = MathEnvironment("my_math", {"difficulty_levels": ["easy"]})
        game_env = GameEnvironment("my_game", {"game_type": "tic_tac_toe"})
        
        # Store them normally
        self.environments["math"] = math_env
        self.environments["game"] = game_env
        
        # Register with gym normally
        for env in self.environments.values():
            self.gym.trajectory_api.register_environment(env)
    
    async def train_agent(self):
        """Normal async method that trains the agent"""
        # Initialize normally
        self.setup_environments()
        
        # Run episodes normally
        for env_name in self.environments:
            trajectory = await self.gym.run_episode(env_name)
            print(f"Completed {env_name}: {trajectory.total_reward:.2f} reward")

# ============================================================================
# NORMAL FUNCTION USAGE
# ============================================================================

def create_math_solver():
    """Normal function that creates a math-solving environment"""
    return MathEnvironment("solver", {
        "difficulty_levels": ["medium"],
        "problem_types": ["algebra", "calculus"]
    })

async def run_training_session(num_episodes=50):
    """Normal async function for training"""
    
    # Create gym normally
    rl_gym = LoomRLGym()
    
    # Add environments normally
    rl_gym.initialize_environments([
        {"type": "math", "id": "training_math", "config": {}},
        {"type": "game", "id": "training_game", "config": {}}
    ])
    
    # Run training normally
    results = []
    for i in range(num_episodes):
        env_id = np.random.choice(list(rl_gym.environments.keys()))
        trajectory = await rl_gym.run_episode(env_id)
        results.append(trajectory.total_reward)
    
    return np.mean(results)

# ============================================================================
# NORMAL INHERITANCE PATTERNS
# ============================================================================

class CustomMathEnvironment(MathEnvironment):
    """Normal inheritance from LoomOS environment"""
    
    def __init__(self, env_id, config=None):
        # Call parent constructor normally
        super().__init__(env_id, config)
        
        # Add custom behavior
        self.custom_scoring = True
    
    def step(self, action):
        """Override parent method normally"""
        # Call parent method
        state, reward, done, info = super().step(action)
        
        # Add custom logic
        if self.custom_scoring and "correct" in str(action).lower():
            reward += 1.0  # Bonus for using "correct" in answer
        
        return state, reward, done, info

# ============================================================================
# NORMAL CONTEXT MANAGER USAGE
# ============================================================================

class GymSession:
    """Normal context manager for gym sessions"""
    
    def __init__(self, env_configs):
        self.env_configs = env_configs
        self.gym = None
    
    async def __aenter__(self):
        # Setup normally
        self.gym = LoomRLGym()
        self.gym.initialize_environments(self.env_configs)
        return self.gym
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cleanup normally
        if self.gym:
            stats = self.gym.get_gym_stats()
            print(f"Session ended. Total episodes: {stats['gym_metrics']['total_episodes']}")

# ============================================================================
# NORMAL DECORATOR PATTERNS
# ============================================================================

def track_performance(func):
    """Normal decorator that tracks RL performance"""
    async def wrapper(*args, **kwargs):
        start_time = asyncio.get_event_loop().time()
        result = await func(*args, **kwargs)
        end_time = asyncio.get_event_loop().time()
        
        print(f"Function {func.__name__} took {end_time - start_time:.2f}s")
        return result
    return wrapper

@track_performance
async def run_benchmark():
    """Normal decorated function"""
    gym = LoomRLGym()
    gym.initialize_environments([
        {"type": "math", "id": "benchmark", "config": {}}
    ])
    
    trajectory = await gym.run_episode("benchmark")
    return trajectory.total_reward

# ============================================================================
# NORMAL CONFIGURATION PATTERNS
# ============================================================================

# Normal dictionary configuration
TRAINING_CONFIG = {
    "environments": [
        {"type": "math", "id": "math_1", "config": {"difficulty_levels": ["easy"]}},
        {"type": "game", "id": "game_1", "config": {"game_type": "tic_tac_toe"}},
    ],
    "training": {
        "episodes_per_update": 100,
        "learning_rate": 3e-4,
        "batch_size": 32
    }
}

# Normal dataclass configuration
from dataclasses import dataclass

@dataclass
class MyTrainingConfig:
    """Normal dataclass for configuration"""
    num_environments: int = 3
    training_episodes: int = 1000
    learning_rate: float = 3e-4
    use_curriculum: bool = True

# ============================================================================
# NORMAL USAGE EXAMPLES
# ============================================================================

async def example_1_basic_usage():
    """Example 1: Basic gym usage like any Python module"""
    
    # Normal imports and instantiation
    gym = LoomRLGym()
    
    # Normal method calls
    gym.initialize_environments([
        {"type": "math", "id": "basic_math", "config": {}}
    ])
    
    # Normal async usage
    trajectory = await gym.run_episode("basic_math")
    
    # Normal property access
    print(f"Episode reward: {trajectory.total_reward}")
    print(f"Episode length: {trajectory.episode_length}")

async def example_2_factory_pattern():
    """Example 2: Using factory patterns normally"""
    
    # Normal factory usage
    from rl.loom_gym import EnvironmentFactory, EnvironmentType
    
    # Create environments normally
    math_env = EnvironmentFactory.create_environment(
        EnvironmentType.MATH, 
        "factory_math", 
        {"difficulty_levels": ["medium"]}
    )
    
    # Use normally
    state = math_env.reset()
    action = "x = 5"
    next_state, reward, done, info = math_env.step(action)
    
    print(f"Factory environment created and used normally!")

async def example_3_integration():
    """Example 3: Full integration usage"""
    
    # Normal integration setup
    system = await create_integrated_rl_system()
    
    # Normal training
    results = await system.run_training_loop(max_iterations=3)
    
    # Normal results access
    stats = system.get_integration_stats()
    print(f"Training completed with {stats['total_episodes_collected']} episodes")

# ============================================================================
# NORMAL MAIN EXECUTION
# ============================================================================

async def main():
    """Normal main function"""
    
    print("🎮 LoomOS RL Gym - Normal Python Module Usage")
    print("=" * 60)
    
    # Run examples normally
    print("\n📝 Example 1: Basic Usage")
    await example_1_basic_usage()
    
    print("\n🏭 Example 2: Factory Pattern")
    await example_2_factory_pattern()
    
    print("\n🔗 Example 3: Integration")
    await example_3_integration()
    
    # Use custom classes normally
    print("\n🤖 Example 4: Custom Agent")
    agent = MyCustomAgent()
    await agent.train_agent()
    
    # Use context manager normally
    print("\n🎯 Example 5: Context Manager")
    async with GymSession(TRAINING_CONFIG["environments"]) as session:
        trajectory = await session.run_episode("math_1")
        print(f"Context manager session: {trajectory.total_reward:.2f} reward")
    
    # Use decorators normally
    print("\n⏱️  Example 6: Decorated Function")
    reward = await run_benchmark()
    print(f"Benchmark reward: {reward:.2f}")
    
    print("\n✅ All examples show normal Python module usage!")

if __name__ == "__main__":
    # Normal execution
    asyncio.run(main())