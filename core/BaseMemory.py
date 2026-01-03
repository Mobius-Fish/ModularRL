from abc import ABC, abstractmethod

class BaseMemory(ABC):
    """职责：存储与采样数据"""
    @abstractmethod
    def push(self, state, action, reward, next_state, done):
        pass
    
    @abstractmethod
    def sample(self, batch_size):
        pass