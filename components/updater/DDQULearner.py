from components.updater.DQULearner import DQULearner
import torch
import torch.nn.functional as F

class DDQULearner(DQULearner): # 继承之前的 Learner
    def update(self, batch_data):
        states, actions, rewards, next_states, dones = batch_data
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # 1. Current Q 保持不变
        curr_q = self.model(states).gather(1, actions)

        # 2. Target Q 改为 Double DQN 逻辑
        with torch.no_grad():
            # --- 核心改动点 ---
            # A: 使用 [当前网络] 选择下一状态的最优动作索引
            next_actions = self.model(next_states).argmax(1).unsqueeze(1)
            # B: 使用 [Target网络] 来评估这个动作的分数
            next_q_values = self.target_model(next_states).gather(1, next_actions)
            
            target_q = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = F.mse_loss(curr_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), curr_q.mean().item()