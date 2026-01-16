from abc import ABC, abstractmethod

class BaseLearner(ABC):
    @abstractmethod
    def update(self, batch_data):
        pass