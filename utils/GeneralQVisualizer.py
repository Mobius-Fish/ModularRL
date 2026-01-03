import numpy as np
import matplotlib.pyplot as plt
import torch
import os

class GeneralQVisualizer:
    """
    通用 Q 值地形绘制器 (支持高维状态切片)。
    """
    def __init__(self, env, device, x_idx=0, y_idx=1, x_range=(-1, 1), y_range=(-1, 1), labels=("X", "Y")):
        self.env = env
        self.device = device
        self.x_idx = x_idx  # 要画的第一个维度索引
        self.y_idx = y_idx  # 要画的第二个维度索引
        self.x_range = x_range # 强制指定的绘图范围
        self.y_range = y_range
        self.labels = labels
        
        # 获取状态的总维度
        self.state_dim = np.prod(env.observation_space.shape)

    def plot(self, model, step, log_dir):
        # 创建网格
        x = np.linspace(self.x_range[0], self.x_range[1], 50)
        y = np.linspace(self.y_range[0], self.y_range[1], 50)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        model.eval()
        with torch.no_grad():
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    # 1. 创建一个全 0 的基准状态向量
                    state = np.zeros(self.state_dim)
                    # 2. 修改我们关心的两个维度
                    state[self.x_idx] = X[i, j]
                    state[self.y_idx] = Y[i, j]
                    
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    # 3. 获取 Max Q
                    Z[i, j] = model(state_t).max().item()
        
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
        ax.set_title(f'Q-Value Surface (Dims {self.x_idx}&{self.y_idx}) at Step {step}')
        ax.set_xlabel(self.labels[0])
        ax.set_ylabel(self.labels[1])
        ax.set_zlabel('Max Q Value')
        fig.colorbar(surf)
        
        save_path = os.path.join(log_dir, f"q_surface_{step}.png")
        plt.savefig(save_path)
        plt.close(fig)