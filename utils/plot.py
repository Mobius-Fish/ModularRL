import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mysql.connector
import json
import numpy as np

# ==========================================
# 1. 数据库配置
# ==========================================
DB_CONFIG = {
    "host": "localhost",
    "user": "rl_learner",
    "password": "rl_pass",
    "database": "rl_experiments"
}

# ==========================================
# 2. 从 params_json 中提取超参数
# ==========================================
def extract_param(params_json, param_path):
    try:
        params = json.loads(params_json)
        value = params
        for k in param_path.split("."):
            value = value[k]
        return value
    except Exception:
        return np.nan


# ==========================================
# 3. 从 metric_events 读取并 pivot 成宽表
# ==========================================
def load_metrics_from_db(
    exp_name,
    env_name,
    algo_names,
    metric_names,
    step_type="episode"
):
    conn = mysql.connector.connect(**DB_CONFIG)

    algo_placeholders = ",".join(["%s"] * len(algo_names))
    metric_placeholders = ",".join(["%s"] * len(metric_names))

    query = f"""
    SELECT
        e.exp_id,
        e.algo_name,
        e.seed,
        e.params_json,
        m.step,
        m.metric_name,
        m.metric_value
    FROM experiments e
    JOIN metric_events m ON e.exp_id = m.exp_id
    WHERE e.exp_name = %s
      AND e.env_name = %s
      AND e.algo_name IN ({algo_placeholders})
      AND m.metric_name IN ({metric_placeholders})
      AND m.step_type = %s
    ORDER BY e.exp_id, m.step
    """

    params = (
        [exp_name, env_name]
        + algo_names
        + metric_names
        + [step_type]
    )

    df = pd.read_sql(query, conn, params=params)
    conn.close()

    if df.empty:
        raise RuntimeError("数据库中未找到匹配的 metric_events 数据")

    # pivot：一行 = (exp_id, step)
    df = df.pivot_table(
        index=["exp_id", "algo_name", "seed", "params_json", "step"],
        columns="metric_name",
        values="metric_value"
    ).reset_index()

    df = df.rename(columns={"step": "episode"})
    return df


# ==========================================
# 4. 按最优超参数过滤
# ==========================================
def filter_by_best_params(df, param_selector):
    mask = pd.Series(False, index=df.index)

    for algo, rules in param_selector.items():
        algo_mask = df["algo_name"] == algo

        for param_path, allowed_values in rules.items():
            values = df.loc[algo_mask, "params_json"].apply(
                lambda x: extract_param(x, param_path)
            )
            algo_mask &= values.isin(allowed_values)

        mask |= algo_mask

    filtered = df[mask].copy()
    if filtered.empty:
        raise RuntimeError("过滤后无数据，请检查 param_selector")

    return filtered


# ==========================================
# 5. 平滑（rolling mean）
# ==========================================
def apply_smoothing(df, metrics, window):
    smoothed = df.groupby("exp_id")[metrics].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )

    out = df.copy()
    for m in metrics:
        out[m] = smoothed[m]
    return out


# ==========================================
# 6. 指标显示配置
# ==========================================
METRIC_CONFIG = {
    "episode_reward": {
        "title": "Episode Reward",
        "ylabel": "Reward"
    },
    "avg_q": {
        "title": "Average Q Value",
        "ylabel": "Q Value"
    },
    "critic_loss": {
        "title": "Critic Loss",
        "ylabel": "Loss"
    },
    "policy_loss": {
        "title": "Policy Loss",
        "ylabel": "Loss"
    },
    "entropy": {
        "title": "Policy Entropy",
        "ylabel": "Entropy"
    }
}


# ==========================================
# 7. 主绘图函数
# ==========================================
def plot_algorithm_comparison(
    exp_name,
    env_name,
    param_selector,
    metric,
    smooth_window=10
):
    if metric not in METRIC_CONFIG:
        raise ValueError(f"不支持的 metric: {metric}")

    algo_names = list(param_selector.keys())

    df = load_metrics_from_db(
        exp_name=exp_name,
        env_name=env_name,
        algo_names=algo_names,
        metric_names=[metric],
        step_type="episode"
    )

    df = filter_by_best_params(df, param_selector)

    if smooth_window > 1:
        df = apply_smoothing(df, [metric], smooth_window)

    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(8, 6))

    sns.lineplot(
        data=df,
        x="episode",
        y=metric,
        hue="algo_name",
        estimator="mean",
        errorbar="sd"
    )

    plt.title(f"{METRIC_CONFIG[metric]['title']} | {env_name}")
    plt.xlabel("Episode")
    plt.ylabel(METRIC_CONFIG[metric]["ylabel"])

    plt.tight_layout()

    save_path = f"./output/{exp_name}_{env_name}_{metric}.png"
    plt.savefig(save_path, dpi=300)
    print(f"图表已保存: {save_path}")
    plt.show()


# ==========================================
# 8. 程序入口
# ==========================================
if __name__ == "__main__":

    best_param_selector = {
        "SAC": {
            "lr": [3e-4],
        }
    }

    plot_algorithm_comparison(
        exp_name="SoftActorCritic",
        env_name="LunarLanderContinuous-v3",
        param_selector=best_param_selector,
        metric="episode_reward",   # 可换成 avg_q / entropy / critic_loss
        smooth_window=20
    )
