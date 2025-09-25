# RL Environments

import gym

class LoomEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(10)

    def reset(self):
        return 0

    def step(self, action):
        return 0, 1, True, {}