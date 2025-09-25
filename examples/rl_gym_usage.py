#!/usr/bin/env python3
"""
LoomOS RL Gym - Simple Usage Examples

This demonstrates how the RL Gym works as a normal Python module
that you can import and use in your projects.
"""

import asyncio
import sys
from pathlib import Path

# Add LoomOS to Python path
sys.path.append(str(Path(__file__).parent))

# Import LoomOS RL Gym components - works like any normal module!
from rl.loom_gym import (
    LoomRLGym, 
    EnvironmentType, 
    EnvironmentFactory,
    MathEnvironment,
    GameEnvironment,
    ToolCallEnvironment
)

from rl.gym_integration import (
    GymPPOIntegration,
    GymTrainingConfig,
    create_integrated_rl_system
)

from rl.ppo_trainer import PPOConfig

async def simple_example():
    """Simple example - just like using any Python module"""
    
    print("🎮 LoomOS RL Gym - Simple Module Usage")
    print("=" * 50)
    
    # 1. Create a single environment (like creating any Python object)
    print("\n📚 Creating a Math Environment...")
    math_env = MathEnvironment("math_1", {
        "difficulty_levels": ["easy"],
        "problem_types": ["algebra"]
    })
    
    # 2. Use it like any gym environment
    print("🔄 Running an episode...")
    state = math_env.reset()
    print(f"Initial state: {state[:100]}...")
    
    # Take some actions
    for step in range(3):
        action = f"Step {step + 1}: 2x = 8"
        next_state, reward, done, info = math_env.step(action)
        print(f"  Step {step + 1}: Reward = {reward:.2f}")
        
        if done:
            print(f"  Episode finished! Success: {info.get('is_correct', False)}")
            break
    
    # 3. Get metrics (normal method call)
    metrics = math_env.get_metrics()
    print(f"\n📊 Environment Metrics: {metrics}")

async def gym_system_example():
    """Example using the complete gym system"""
    
    print("\n🏟️  Complete Gym System Example")
    print("=" * 50)
    
    # 1. Create gym system (normal instantiation)
    gym = LoomRLGym()
    
    # 2. Initialize with environments (normal method call)
    env_configs = [
        {"type": "math", "id": "math_easy", "config": {"difficulty_levels": ["easy"]}},
        {"type": "game", "id": "tictactoe", "config": {"game_type": "tic_tac_toe"}},
        {"type": "toolcall", "id": "calculator", "config": {"tools": ["calculator"]}}
    ]
    gym.initialize_environments(env_configs)
    
    print(f"✅ Initialized {len(gym.environments)} environments")
    
    # 3. Run episodes (async method calls)
    print("\n🎯 Running sample episodes...")
    for env_id in list(gym.environments.keys()):
        trajectory = await gym.run_episode(env_id)
        print(f"  {env_id}: {trajectory.episode_length} steps, reward: {trajectory.total_reward:.2f}")
    
    # 4. Get statistics (normal property access)
    stats = gym.get_gym_stats()
    print(f"\n📈 Total episodes run: {stats['gym_metrics']['total_episodes']}")

async def integrated_training_example():
    """Example of integrated training system"""
    
    print("\n🚀 Integrated Training Example")
    print("=" * 50)
    
    # 1. Configure training (normal dataclass creation)
    gym_config = GymTrainingConfig(
        environments_per_type=2,
        collect_episodes_per_update=10,
        enabled_env_types=[EnvironmentType.MATH, EnvironmentType.GAME]
    )
    
    ppo_config = PPOConfig(
        total_episodes=100,
        batch_size=16,
        learning_rate=3e-4,
        use_wandb=False  # Disable for demo
    )
    
    # 2. Create integrated system (factory function)
    print("🔧 Creating integrated RL system...")
    integration = await create_integrated_rl_system(gym_config, ppo_config)
    
    # 3. Run training (simple method call)
    print("🎓 Running training for 3 iterations...")
    results = await integration.run_training_loop(max_iterations=3)
    
    # 4. Get final stats (normal method call)
    final_stats = integration.get_integration_stats()
    print(f"✅ Training complete!")
    print(f"   Episodes collected: {final_stats['total_episodes_collected']}")
    print(f"   Training iterations: {final_stats['training_iterations']}")

def factory_example():
    """Example using the environment factory"""
    
    print("\n🏭 Environment Factory Example")
    print("=" * 50)
    
    # 1. Create individual environments (static method calls)
    math_env = EnvironmentFactory.create_environment(
        EnvironmentType.MATH, 
        "factory_math", 
        {"difficulty_levels": ["medium"]}
    )
    
    game_env = EnvironmentFactory.create_environment(
        EnvironmentType.GAME,
        "factory_game",
        {"game_type": "tic_tac_toe"}
    )
    
    print(f"✅ Created {math_env.env_id} ({math_env.env_type.value})")
    print(f"✅ Created {game_env.env_id} ({game_env.env_type.value})")
    
    # 2. Create a complete suite (factory method)
    env_suite = EnvironmentFactory.create_environment_suite()
    print(f"✅ Created complete suite with {len(env_suite)} environments")
    
    # 3. Use them normally
    for env in env_suite[:2]:  # Just test first 2
        print(f"   {env.env_id}: Status = {env.get_status().value}")

def import_examples():
    """Show different ways to import and use the module"""
    
    print("\n📦 Import Examples")
    print("=" * 50)
    
    # You can import specific components
    print("✅ from rl.loom_gym import LoomRLGym")
    print("✅ from rl.loom_gym import MathEnvironment, GameEnvironment")
    print("✅ from rl.gym_integration import create_integrated_rl_system")
    
    # Or import the whole module
    print("✅ import rl.loom_gym as gym")
    print("✅ import rl.gym_integration as integration")
    
    # Works with normal Python patterns
    print("✅ Normal instantiation: gym = LoomRLGym()")
    print("✅ Normal method calls: await gym.run_episode('env_1')")
    print("✅ Normal properties: gym.environments, gym.metrics")
    print("✅ Normal inheritance: class CustomEnv(LoomEnvironment)")

async def main():
    """Run all examples"""
    
    print("🎯 LoomOS RL Gym - Normal Python Module Usage")
    print("=" * 60)
    
    # Show import patterns
    import_examples()
    
    # Simple single environment usage
    await simple_example()
    
    # Complete gym system usage
    await gym_system_example()
    
    # Factory pattern usage
    factory_example()
    
    # Integrated training usage
    await integrated_training_example()
    
    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("🎉 LoomOS RL Gym works like any normal Python module!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())