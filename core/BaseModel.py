from abc import ABC, abstractmethod
import torch.nn as nn

class BaseModel(nn.Module):
    """职责：表征(Representation)，输入状态，输出数值"""
    pass