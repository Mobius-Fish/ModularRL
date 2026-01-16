import torch
import torch.nn.functional as F

class SDQULearner:
    """
    职责：计算 Loss 并更新网络参数。
    这里包含了 Target Policy 的概念。
    """
    def __init__(self, model, target_model, optimizer, gamma, device):
        self.model = model
        self.target_model = target_model
        self.optimizer = optimizer
        self.gamma = gamma
        self.device = device
        
        # 硬更新：初始化时同步参数
        self.target_model.load_state_dict(self.model.state_dict())

    def update(self, batch_data):
        states, actions, rewards, next_states, dones = batch_data
        
        # 转换为 Tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # 1. 计算 Current Q (Behavior Policy 产生的数据)
        # gather(1, actions) 提取出我们实际执行动作对应的 Q 值
        curr_q = self.model(states).gather(1, actions)

        # 2. 计算 Target Q (Target Policy - Greedy)
        with torch.no_grad():
            # 这里进行了修改，使用了 Softmax 代替了原本 的 max 操作，实现了 softmax DQN
            next_q_values = self.target_model(next_states)
            tau = 5.0  # 温度参数，可以调整软化程度，这是一个超参数，由于这里是尝试，就不写在config里了，根据论文里说 5.0 不错
            weights = torch.softmax(tau * next_q_values, dim=1) # 计算 softmax 权重，相当于原本使用max求出来的东西
            next_v = torch.sum(weights * next_q_values, dim=1, keepdim=True)
            target_q = rewards + (self.gamma * next_v * (1 - dones))

        # 3. 计算 Loss
        loss = F.mse_loss(curr_q, target_q)

        # 4. 梯度更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), curr_q.mean().item()
    
    def sync_target_network(self):
        """将当前网络的权重复制给 Target 网络"""
        self.target_model.load_state_dict(self.model.state_dict())