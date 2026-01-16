from core.BaseModel import BaseModel
import torch.nn as nn
import torch

class ContinuousQNetwork(BaseModel):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # 输入是 state 和 action 的拼接
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # 输出标量 Q 值
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)