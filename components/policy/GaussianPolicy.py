from torch.distributions import Normal
import torch.nn as nn
import torch

class GaussianPolicy(nn.Module):
    """
    用于连续动作空间的高斯策略网络
    输入: state
    输出: action 的均值和对数标准差 (log_std)
    """
    def __init__(self, state_dim, action_dim, action_high=1.0, device='cpu'):
        super().__init__()
        self.action_high = action_high
        self.device = device

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        
        # 限制 log_std 范围，防止训练不稳定
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        
        # 重参数化采样: z = mu + sigma * epsilon
        x_t = normal.rsample()  
        
        # Tanh Squashing: 将动作限制在 [-1, 1] 之间
        y_t = torch.tanh(x_t)
        action = y_t * self.action_high
        
        # 计算 log_prob (需要修正 tanh 带来的分布变化)
        # 公式: log_prob(a) = log_prob(u) - log(1 - tanh(u)^2)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_high * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        return action, log_prob

    def get_action(self, state):
        # 这是一个兼容接口，用于 Runner
        self.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, _ = self.sample(state_t)
        return action.cpu().numpy()[0]