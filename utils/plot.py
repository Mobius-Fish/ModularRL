import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mysql.connector
import numpy as np

# ==========================================
# 1. 数据库配置 (需与你训练时的配置一致)
# ==========================================
DB_CONFIG = {
    'host': "localhost",
    'user': "rl_learner",   # 替换为你的用户名
    'password': "rl_pass",  # 替换为你的密码
    'database': "rl_experiments"
}

# ==========================================
# 2. 数据获取与处理函数
# ==========================================
def get_data_from_db(env_name="CartPole-v1"):
    """
    从数据库中提取指定环境的所有实验数据。
    """
    conn = mysql.connector.connect(**DB_CONFIG)
    
    # 联表查询：获取实验配置(lr)和训练日志
    # 注意：这里我们提取 lr 作为 hue (分类依据)
    query = f"""
    SELECT e.exp_id, e.lr, e.seed, 
           l.episode, l.episode_reward, l.avg_loss, l.avg_q
    FROM experiments e
    JOIN training_logs l ON e.exp_id = l.exp_id
    WHERE e.env_name = '{env_name}'
    """
    
    print(f"正在从数据库读取 {env_name} 的数据...")
    df = pd.read_sql(query, conn)
    conn.close()
    
    if df.empty:
        print("警告：数据库中没有找到相关数据！")
        return None

    # 数据预处理：将 lr 转换为字符串，这样 seaborn 会把它当作离散类别而不是连续数值
    df['Learning Rate'] = df['lr'].astype(str)
    
    return df

def apply_smoothing(df, window=10):
    """
    对数据进行滑动窗口平滑处理，使曲线更美观。
    注意：为了保持 Seaborn 的多 Seed 统计特性，我们需要对每个 Seed 单独平滑。
    """
    # 按 exp_id (即单个实验) 分组，然后对指标列进行 Rolling Mean
    metrics = ['episode_reward', 'avg_loss', 'avg_q']
    
    # 使用 lambda 保持数据对齐
    smoothed_df = df.groupby('exp_id')[metrics].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
    
    # 将平滑后的数据覆盖回原 DataFrame (或者创建新列)
    df_clean = df.copy()
    for metric in metrics:
        df_clean[metric] = smoothed_df[metric]
        
    return df_clean

# ==========================================
# 3. 核心绘图函数
# ==========================================
def plot_all_metrics(env_name="CartPole-v1", smooth_window=10):
    # 1. 获取数据
    df = get_data_from_db(env_name)
    if df is None:
        return

    # 2. 平滑数据 (RL 数据通常噪点很大，平滑后更容易看清趋势)
    if smooth_window > 1:
        df = apply_smoothing(df, window=smooth_window)

    # 3. 设置 Seaborn 风格
    sns.set_theme(style="darkgrid")
    
    # 4. 创建画布：1行3列
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # --- 图 1: Reward (最重要) ---
    sns.lineplot(
        data=df, 
        x="episode", 
        y="episode_reward", 
        hue="Learning Rate", 
        estimator='mean',  # 绘制平均值
        errorbar='sd',     # 绘制标准差阴影 (Standard Deviation)
        ax=axes[0]
    )
    axes[0].set_title(f"Episode Reward ({env_name})")
    axes[0].set_ylabel("Reward")
    axes[0].set_xlabel("Episode")

    # --- 图 2: Loss (监控学习过程) ---
    sns.lineplot(
        data=df, 
        x="episode", 
        y="avg_loss", 
        hue="Learning Rate", 
        estimator='mean',
        errorbar='sd',
        ax=axes[1]
    )
    axes[1].set_title("Training Loss")
    axes[1].set_ylabel("Loss (MSE)")
    axes[1].set_xlabel("Episode")
    # Loss 通常变化幅度大，可以用对数坐标 (可选)
    # axes[1].set_yscale('log') 

    # --- 图 3: Mean Q-Value (监控高估问题) ---
    sns.lineplot(
        data=df, 
        x="episode", 
        y="avg_q", 
        hue="Learning Rate", 
        estimator='mean',
        errorbar='sd',
        ax=axes[2]
    )
    axes[2].set_title("Average Q-Value")
    axes[2].set_ylabel("Q Value")
    axes[2].set_xlabel("Episode")

    # 5. 调整布局并保存
    plt.tight_layout()
    plt.savefig(f"./output/{env_name}_analysis.png", dpi=300)
    print(f"图表已保存为 {env_name}_analysis.png")
    plt.show()

# ==========================================
# 主入口
# ==========================================
if __name__ == "__main__":
    # 这里的 env_name 必须和你数据库里存的一致
    plot_all_metrics(env_name="CartPole-v1", smooth_window=5)