import gymnasium as gym
import torch
import torch.optim as optim
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from components.policy.DiscretePolicy import DiscretePolicy
from components.strategy.EpsilonGreedy import EpsilonGreedy
from components.memory.ReplayBuffer import ReplayBuffer
from components.updater.DQULearner import DQULearner
from utils.GeneralQVisualizer import GeneralQVisualizer
from core.BaseModel import BaseModel
from components.representation.QNetwork import QNetwork
import os
import random
import hydra
from omegaconf import DictConfig, OmegaConf
from utils.ExperimentDB import ExperimentDB
from collections import deque

class Runner:
    def __init__(self, config):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)

        # 1. 初始化环境
        self.env = gym.make(self.cfg.env_name)
        
        # 2. 初始化 TensorBoard
        run_name = f"{self.cfg.env_name}_DQN_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.writer = SummaryWriter(f"runs/{run_name}")
        self.vis_dir = f"runs/{run_name}/plots"
        os.makedirs(self.vis_dir, exist_ok=True)
        
        # 3. 初始化 Representation (Model)
        state_dim = np.prod(self.env.observation_space.shape)
        action_dim = self.env.action_space.n
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device) # Target Policy 用
        
        # 4. 初始化组件
        self.policy = DiscretePolicy(self.q_net, self.device)
        self.explorer = EpsilonGreedy(decay=config['epsilon_decay'])
        self.memory = ReplayBuffer(config['buffer_size'])
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config['lr'])
        
        # 5. 初始化 Learner
        self.learner = DQULearner(
            model=self.q_net,
            target_model=self.target_net,
            optimizer=self.optimizer,
            gamma=config['gamma'],
            device=self.device
        )

        # 6. 初始化日志记录器
        self.db = ExperimentDB(self.cfg.db_config) if "db_config" in self.cfg else None
    
    def run(self):
        print(f"🚀 Start Training on {self.device}...")
        exp_id = self.db.start_new_experiment(self.cfg)
        state, _ = self.env.reset()
        global_step = 0
        all_episode_rewards = []
        
        for episode in range(self.cfg['max_episodes']):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_losses = []
            episode_qs = []
            done = False
            
            while not done:
                global_step += 1
                
                # --- A. Agent 决策 (Behavior Policy + Exploration) ---
                # 1. Policy 给出建议
                policy_action, _ = self.policy.get_action(state)
                # 2. Exploration 进行修饰
                action, epsilon = self.explorer.select_action(
                    policy_action, self.env.action_space, global_step
                )
                
                # --- B. 环境交互 ---
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # --- C. 存入记忆 ---
                self.memory.push(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                
                # --- D. 学习 (Updater) ---
                if len(self.memory) > self.cfg['batch_size']:
                    batch_data = self.memory.sample(self.cfg['batch_size'])
                    loss, mean_q = self.learner.update(batch_data)
                    
                    # 记录训练数据(TensorBoard)
                    if global_step % 100 == 0:
                        self.writer.add_scalar("losses/td_loss", loss, global_step)
                        self.writer.add_scalar("charts/mean_q", mean_q, global_step)
                        self.writer.add_scalar("charts/epsilon", epsilon, global_step)
                    # mysql
                    episode_losses.append(loss)
                    episode_qs.append(mean_q)

                # --- E. Target Network 更新 ---
                if global_step % self.cfg['target_update_freq'] == 0:
                    self.learner.sync_target_network()
            
            # 1. 回合结束，计算本回合平均指标
            avg_loss = np.mean(episode_losses) if episode_losses else 0
            avg_q = np.mean(episode_qs) if episode_qs else 0
            all_episode_rewards.append(episode_reward)

            # 2. 存入数据库：每轮记录一次
            self.db.log_step_data(episode, global_step, episode_reward, avg_loss, avg_q)

            # --- Episode 结束后的记录与可视化 ---
            if (episode + 1) % 20 == 0:
                self.writer.add_scalar("charts/episode_reward", episode_reward, global_step)
                print(f"Episode {episode+1} | Step {global_step} | Reward: {episode_reward:.2f} | Epsilon: {epsilon:.3f}")
            
            # 每 100 个 Episode 画一次 Q 值地形图
            if (episode + 1) % 100 == 0:
                self.visualizer.plot(self.q_net, global_step, self.vis_dir)

        # 训练结束，更新最终状态到数据库
        final_reward = np.mean(all_episode_rewards[-10:]) # 最后10次平均
        self.db.update_final_status(final_reward, global_step)

        # 结束
        self.env.close()
        self.writer.close()
        print("🎉 Training Finished!")