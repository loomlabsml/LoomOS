#!/usr/bin/env python3
"""
Quick Demo: LoomOS RL Gym as Normal Python Module

Run this to see how LoomOS RL Gym works just like any standard Python package!
"""

import asyncio

# Import LoomOS components just like any normal Python module
from rl.loom_gym import LoomRLGym, EnvironmentType, MathEnvironment
from rl.gym_integration import GymTrainingConfig, create_integrated_rl_system

async def quick_demo():
    """Quick demonstration of normal module usage"""
    
    print("🎮 LoomOS RL Gym - Quick Demo")
    print("=" * 40)
    
    # 1. NORMAL OBJECT CREATION
    print("\n1️⃣  Creating environment (like any Python class):")
    math_env = MathEnvironment("demo_math", {
        "difficulty_levels": ["easy"],
        "problem_types": ["algebra"]
    })
    print(f"   ✅ Created: {math_env.env_id}")
    
    # 2. NORMAL METHOD CALLS
    print("\n2️⃣  Running episode (normal method calls):")
    state = math_env.reset()
    print(f"   📋 Initial state: {state[:50]}...")
    
    action = "2x + 5 = 13"
    next_state, reward, done, info = math_env.step(action)
    print(f"   🎯 Action: {action}")
    print(f"   💰 Reward: {reward}")
    print(f"   ✅ Done: {done}")
    
    # 3. NORMAL PROPERTY ACCESS
    print("\n3️⃣  Getting metrics (normal property access):")
    metrics = math_env.get_metrics()
    status = math_env.get_status()
    print(f"   📊 Metrics: {metrics}")
    print(f"   📈 Status: {status.value}")
    
    # 4. NORMAL ASYNC USAGE
    print("\n4️⃣  Using async features (normal async/await):")
    gym = LoomRLGym()
    gym.initialize_environments([
        {"type": "math", "id": "async_math", "config": {"difficulty_levels": ["easy"]}}
    ])
    
    trajectory = await gym.run_episode("async_math")
    print(f"   🎯 Episode steps: {trajectory.episode_length}")
    print(f"   💰 Total reward: {trajectory.total_reward:.2f}")
    
    # 5. NORMAL FACTORY PATTERNS
    print("\n5️⃣  Factory pattern (normal static methods):")
    config = GymTrainingConfig(environments_per_type=1)
    print(f"   ⚙️  Config created: {config.enabled_env_types}")
    
    print("\n✅ Demo complete - works like any normal Python module!")

if __name__ == "__main__":
    # Normal Python execution
    asyncio.run(quick_demo())