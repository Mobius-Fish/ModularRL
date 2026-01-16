import torch
import torch.nn.functional as F
from core.BaseLearner import BaseLearner

class SACLearner(BaseLearner):
    def __init__(self, q1, q2, policy, target_q1, target_q2, 
                 q_optimizer, policy_optimizer, gamma, alpha, device):
        # 使用 Double Q Trick (两个 Q 网络) 来减少高估，这是现代标配
        self.q1 = q1
        self.q2 = q2
        self.policy = policy
        self.target_q1 = target_q1
        self.target_q2 = target_q2
        
        self.q_optimizer = q_optimizer
        self.policy_optimizer = policy_optimizer
        
        self.gamma = gamma
        self.alpha = alpha  # 熵系数 (Temperature)
        self.device = device
        
        # 初始化 Target 网络
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

    def update(self, batch_data):
        states, actions, rewards, next_states, dones = batch_data
        
        # 转 Tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device) # 注意: 连续动作是 Float
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # ----------------------------
        # 1. 更新 Critic (Q 函数)
        # ----------------------------
        with torch.no_grad():
            # 计算下一个状态的动作和熵
            next_actions, next_log_prob = self.policy.sample(next_states)
            
            # 计算 Target Q 值 (Soft Q 的核心)
            # V(s') = Q(s', a') - alpha * log_prob(a'|s')
            target_q1_val = self.target_q1(next_states, next_actions)
            target_q2_val = self.target_q2(next_states, next_actions)
            min_target_q = torch.min(target_q1_val, target_q2_val)
            
            # Soft Bellman Target
            target_value = min_target_q - self.alpha * next_log_prob
            q_target = rewards + (1 - dones) * self.gamma * target_value

        # 计算当前的 Q 值
        q1_val = self.q1(states, actions)
        q2_val = self.q2(states, actions)
        
        q_loss = F.mse_loss(q1_val, q_target) + F.mse_loss(q2_val, q_target)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # ----------------------------
        # 2. 更新 Actor (Policy)
        # ----------------------------
        # 我们希望 Policy 产生的动作能够获得高 Q 值，且 log_prob 越小越好（高熵）
        new_actions, log_prob = self.policy.sample(states)
        
        q1_new = self.q1(states, new_actions)
        q2_new = self.q2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        # Loss = alpha * log_prob - Q (最小化 Loss 等于 最大化 Q - alpha * log_prob)
        policy_loss = (self.alpha * log_prob - q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        return q_loss.item(), q_new.mean().item()

    def sync_target_network(self, tau=0.005):
        # 软更新 (Soft Update)
        for param, target_param in zip(self.q1.parameters(), self.target_q1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.target_q2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)