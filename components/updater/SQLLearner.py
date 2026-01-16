import torch
import torch.nn.functional as F
from core.BaseLearner import BaseLearner
import numpy as np

class SQLLearner(BaseLearner):
    def __init__(self, q1, policy, target_q1, q_optim, policy_optim, 
                 gamma=0.99, alpha=0.2, device='cpu'):
        # SQL 原文通常只用一个 Q 网络，但为了公平对比也可以用两个。这里简化用一个。
        self.q1 = q1
        self.target_q1 = target_q1
        self.policy = policy
        self.q_optim = q_optim
        self.policy_optim = policy_optim
        self.gamma, self.alpha, self.device = gamma, alpha, device

    def rbf_kernel(self, input_1, input_2, h=None):
        """计算 RBF 核矩阵及其梯度"""
        k_sq_dist = torch.cdist(input_1, input_2) ** 2
        if h is None: # 启发式带宽选择: 中位数技巧
            h = torch.median(k_sq_dist) / (2 * np.log(input_1.size(0) + 1))
        
        # 核矩阵 K(x, y)
        k_matrix = torch.exp(-k_sq_dist / h)
        return k_matrix

    def update(self, batch):
        states, actions, rewards, next_states, dones = [x.to(self.device) for x in batch]

        # --- 1. Critic Update (与 SAC 类似但 Value 计算略有不同) ---
        with torch.no_grad():
            # SQL 使用 Importance Sampling 或直接采样来估计 V_soft
            # V(s') approx Q(s', a') - alpha * log_prob(a')
            # 实际上现代 SQL 实现中，Critic Update 常和 SAC 保持一致
            next_actions = self.policy.sample(next_states)
            target_q_next = self.target_q1(next_states, next_actions)
            # 简化版：假设采样分布接近能量分布，略去 LogProb 项
            # 严格版 SQL 比较复杂，这里采用 Amortized SQL 的简化形式
            v_next = target_q_next 
            q_target = rewards + (1 - dones) * self.gamma * v_next

        q_loss = F.mse_loss(self.q1(states, actions), q_target)
        self.q_optim.zero_grad()
        q_loss.backward()
        self.q_optim.step()

        # --- 2. Actor Update (SQL 的核心: SVGD) ---
        # 我们需要对同一个状态采样多个粒子来计算排斥力
        # 为了简化，我们直接使用 Batch 中的不同样本作为粒子集合
        
        current_actions = self.policy.sample(states)
        current_actions.requires_grad_(True) # 我们需要对 Action 求导

        # 计算 Q 值对 Action 的梯度 (Driving Force)
        q_val = self.q1(states, current_actions)
        q_grad = torch.autograd.grad(q_val.sum(), current_actions)[0]

        # 计算 Kernel Matrix (Repulsive Force)
        # K(a_i, a_j)
        k_matrix = self.rbf_kernel(current_actions, current_actions)
        
        # 计算 Kernel 对 Action 的梯度
        # 这一步计算比较 tricky，公式: sum_j [ k(a_j, a_i) * grad_Q(a_j) + alpha * grad_a_j k(a_j, a_i) ]
        # grad_a_j k(a_j, a_i) = k(a_j, a_i) * 2 * (a_i - a_j) / h
        
        k_sum = k_matrix.sum(dim=1, keepdim=True) # sum_j k(a_j, a_i)
        
        # Stein Variational Gradient 公式
        # Delta a_i = E [ k(a, a') * grad Q(a') / alpha + grad k(a, a') ]
        
        # 第一项: Kernel 加权的 Q 梯度
        term1 = (k_matrix.matmul(q_grad)) / self.alpha
        
        # 第二项: 排斥力 (Entropy)
        # 这里的梯度计算通过自动微分实现有点绕，手动推导更直接
        # grad_k = - (x_i - x_j) / h * K_ij
        # 这里简化处理：通常使用 Autograd 计算 kernel 梯度
        grad_k = -torch.autograd.grad(k_matrix.sum(), current_actions)[0]

        svgd_target_grad = (term1 + grad_k) / states.shape[0]

        # --- 通过 Chain Rule 更新 Policy 网络 ---
        # 我们不能直接把 action 改了，我们要改 Policy 的参数
        # 做法：Action_New = Action_Old + epsilon * svgd_grad
        # Loss = MSE(Policy_Output, Action_New.detach())
        # 或者直接 backward(-svgd_target_grad)
        
        self.policy_optim.zero_grad()
        current_actions.backward(gradient=-svgd_target_grad) # 负梯度最大化
        self.policy_optim.step()
        
        return q_loss.item(), 0.0 # SVGD 没有显示的 Policy Loss 值
    
    def sync_target_network(self, tau=0.005):
        # 软更新 (Soft Update)
        for param, target_param in zip(self.q1.parameters(), self.target_q1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)