import mysql.connector

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
                seed INT,
                lr FLOAT,
                total_steps INT,
                final_avg_reward FLOAT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 表2：过程日志 (每回合/采样步 存一行)
        # 用 exp_id 关联表1，方便联表查询
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_logs (
                log_id INT AUTO_INCREMENT PRIMARY KEY,
                exp_id INT,
                episode INT,
                global_step INT,
                episode_reward FLOAT,
                avg_loss FLOAT,
                avg_q FLOAT,
                FOREIGN KEY (exp_id) REFERENCES experiments(exp_id)
            )
        """)
        self.conn.commit()
        self.current_exp_id = None

    def start_new_experiment(self, cfg):
        sql = "INSERT INTO experiments (exp_name, env_name, seed, lr) VALUES (%s, %s, %s, %s)"
        self.cursor.execute(sql, ("DQN_Modular", cfg.env_name, cfg.seed, cfg.lr))
        self.conn.commit()
        self.current_exp_id = self.cursor.lastrowid
        return self.current_exp_id

    def log_step_data(self, episode, step, reward, loss, q):
        # 记录每回合的详细数据
        sql = """INSERT INTO training_logs (exp_id, episode, global_step, episode_reward, avg_loss, avg_q) 
                 VALUES (%s, %s, %s, %s, %s, %s)"""
        self.cursor.execute(sql, (self.current_exp_id, episode, step, reward, loss, q))
        self.conn.commit()

    def update_final_status(self, final_reward, total_steps):
        sql = "UPDATE experiments SET final_avg_reward = %s, total_steps = %s WHERE exp_id = %s"
        self.cursor.execute(sql, (final_reward, total_steps, self.current_exp_id))
        self.conn.commit()