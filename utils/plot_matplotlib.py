import mysql.connector
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Any

# ==========================================
# 1. DB 配置
# ==========================================
DB_CONFIG = {
    "host": "localhost",
    "user": "rl_learner",
    "password": "rl_pass",
    "database": "rl_experiments"
}

# ==========================================
# 2. 通用工具：解析嵌套超参数
# ==========================================
def extract_param(params_json: str, param_path: str) -> Any:
    """
    param_path 示例:
      - "lr"
      - "optimizer.lr"
      - "model.hidden_dim"
    """
    params = json.loads(params_json)
    keys = param_path.split(".")

    value = params
    for k in keys:
        if not isinstance(value, dict) or k not in value:
            return None
        value = value[k]
    return value


# ==========================================
# 3. 获取数据
# ==========================================
def load_data(env_name: str, param_path: str) -> pd.DataFrame:
    conn = mysql.connector.connect(**DB_CONFIG)

    query = """
    SELECT
        e.exp_id,
        e.seed,
        e.params_json,
        l.episode,
        l.episode_reward,
        l.avg_loss,
        l.avg_q
    FROM experiments e
    JOIN training_logs l ON e.exp_id = l.exp_id
    WHERE e.env_name = %s
    ORDER BY l.episode
    """

    df = pd.read_sql(query, conn, params=(env_name,))
    conn.close()

    if df.empty:
        raise RuntimeError("数据库中没有匹配的数据")

    # 解析超参数
    df["hue_param"] = df["params_json"].apply(
        lambda x: extract_param(x, param_path)
    )

    df = df.dropna(subset=["hue_param"])
    df["hue_param"] = df["hue_param"].astype(str)

    return df


# ==========================================
# 4. 平滑（按 exp_id）
# ==========================================
def smooth_by_experiment(df: pd.DataFrame, metric: str, window: int):
    df = df.copy()
    df[metric] = (
        df.groupby("exp_id")[metric]
        .transform(lambda x: x.rolling(window, min_periods=1).mean())
    )
    return df


# ==========================================
# 5. NeurIPS / ICML 风格绘图
# ==========================================
def plot_metric(
    env_name: str,
    metric: str,
    param_path: str,
    smooth_window: int = 10
):
    """
    metric:
      - episode_reward
      - avg_loss
      - avg_q
    """

    df = load_data(env_name, param_path)

    if smooth_window > 1:
        df = smooth_by_experiment(df, metric, smooth_window)

    # ========= 聚合统计 =========
    stats = (
        df.groupby(["episode", "hue_param"])[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    stats["sem"] = stats["std"] / np.sqrt(stats["count"])
    stats["ci95"] = 1.96 * stats["sem"]

    # ========= Matplotlib 风格 =========
    plt.figure(figsize=(7, 5))
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "axes.linewidth": 1.2
    })

    for key, g in stats.groupby("hue_param"):
        plt.plot(
            g["episode"],
            g["mean"],
            linewidth=2.5,
            label=f"{param_path}={key}"
        )
        plt.fill_between(
            g["episode"],
            g["mean"] - g["ci95"],
            g["mean"] + g["ci95"],
            alpha=0.2
        )

    plt.xlabel("Episode")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f"{env_name}")

    plt.legend(frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = f"./output/{env_name}_{metric}_{param_path.replace('.', '_')}.png"
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"已保存：{save_path}")


# ==========================================
# 6. 示例入口
# ==========================================
if __name__ == "__main__":
    plot_metric(
        env_name="LunarLander-v3",
        metric="episode_reward",
        param_path="optimizer.lr",
        smooth_window=10
    )
