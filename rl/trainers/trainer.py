# Trainer

from ..algos.ppo import PPO

class Trainer:
    def __init__(self):
        self.algo = PPO()

    def train(self, env):
        self.algo.train(env)