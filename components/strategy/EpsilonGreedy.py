from core.BaseExploration import BaseExploration
import numpy as np
import random

class EpsilonGreedy(BaseExploration):
    def __init__(self, start=1.0, end=0.05, decay=2000):
        self.start = start
        self.end = end
        self.decay = decay
    
    def select_action(self, policy_action, action_space, current_step):
        # 计算当前的 epsilon
        epsilon = self.end + (self.start - self.end) * \
                  np.exp(-1. * current_step / self.decay)
        
        if random.random() < epsilon:
            return action_space.sample(), epsilon # 探索
        else:
            return policy_action, epsilon         # 利用