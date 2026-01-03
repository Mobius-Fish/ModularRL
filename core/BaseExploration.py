from abc import ABC, abstractmethod


class BaseExploration(ABC):
    """职责：决定如何在利用(Exploit)与探索(Explore)之间平衡"""
    @abstractmethod
    def select_action(self, policy_action, action_space, current_step):
        pass