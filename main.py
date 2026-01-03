from Runner import Runner
from utils.GeneralQVisualizer import GeneralQVisualizer

if __name__ == "__main__":
    # 配置超参数
    config = {
        'env_name': "LunarLander-v3",
        'lr': 1e-3,              # 稍微降低学习率，防止震荡
        'gamma': 0.99,
        'buffer_size': 50000,    # 需要更大的 Buffer
        'batch_size': 128,
        'epsilon_decay': 50000,  # 需要更长时间的探索！这很重要！
        'target_update_freq': 2000, 
        'max_episodes': 1500     # 至少需要跑 1000 回合才能看到效果
    }

    runner = Runner(config)

runner.visualizer = GeneralQVisualizer(
        runner.env, runner.device, 
        x_idx=0, y_idx=1, 
        x_range=(-0.5, 0.5), y_range=(0, 1.5),
        labels=("X Position (Horizontal)", "Y Position (Height)")
    )
    
runner.run()