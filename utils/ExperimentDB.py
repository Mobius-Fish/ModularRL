import mysql.connector
import json
from omegaconf import OmegaConf

class ExperimentDB:
    def __init__(self, db_cfg):
        self.conn = mysql.connector.connect(
            host=db_cfg.host,
            user=db_cfg.user,
            password=db_cfg.password,
            database=db_cfg.database
        )
        self.cursor = self.conn.cursor()

# 表1：实验总览 (每个 Seed 存一行)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                exp_id INT AUTO_INCREMENT PRIMARY KEY,
                exp_name VARCHAR(100),
                env_name VARCHAR(50),
                algo_name VARCHAR(50),   -- 算法名称 (新加)
                seed INT,
                params_json JSON,        -- 存储所有超参数的 JSON 对象 (新加)
                total_steps INT,
                final_avg_reward FLOAT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 表2：过程日志 (每回合/采样步 存一行)
        # 用 exp_id 关联表1，方便联表查询
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS metric_events (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                exp_id INT NOT NULL,
                step_type ENUM('step', 'episode') NOT NULL,
                step INT NOT NULL,
                metric_name VARCHAR(100) NOT NULL,
                metric_value FLOAT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_exp_metric (exp_id, metric_name),
                FOREIGN KEY (exp_id) REFERENCES experiments(exp_id)
            )
        """)
        self.conn.commit()
        self.current_exp_id = None

    # ---------- Experiment lifecycle ----------
    def start_new_experiment(self, cfg):
        params_dict = OmegaConf.to_container(cfg, resolve=True)
        params_json = json.dumps(params_dict)
        
        sql = "INSERT INTO experiments (exp_name, env_name, algo_name, seed, params_json) VALUES (%s, %s, %s, %s, %s)"
        self.cursor.execute(sql, (cfg.exp_name, cfg.env_name, cfg.algo_name, cfg.seed, params_json))
        self.conn.commit()
        self.current_exp_id = self.cursor.lastrowid
        return self.current_exp_id

    def update_final_status(self, final_reward, total_steps):
        sql = "UPDATE experiments SET final_avg_reward = %s, total_steps = %s WHERE exp_id = %s"
        self.cursor.execute(sql, (final_reward, total_steps, self.current_exp_id))
        self.conn.commit()

    # ---------- Metric logging  ----------
    def log_metrics(self, *, step_type, step, metrics: dict):
        """
        非侵入式指标记录接口

        参数:
        - step_type: 'step' | 'episode'
        - step: global_step 或 episode
        - metrics: dict[str, float]
        """
        assert self.current_exp_id is not None

        rows = [
            (self.current_exp_id, step_type, step, k, float(v))
            for k, v in metrics.items()
            if v is not None
        ]

        sql = """
            INSERT INTO metric_events
            (exp_id, step_type, step, metric_name, metric_value)
            VALUES (%s, %s, %s, %s, %s)
        """
        self.cursor.executemany(sql, rows)
        self.conn.commit()