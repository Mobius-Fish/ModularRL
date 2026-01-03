from core.BaseModel import BaseModel
import torch.nn as nn

class QNetwork(BaseModel):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # 简单的 MLP 结构
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)