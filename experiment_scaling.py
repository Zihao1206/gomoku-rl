"""
实验：放大棋盘，看表格型 Q-Learning 怎么撞上"天花板"。

对几种棋盘各训练【相同局数】，对比：
  · Q 表条目数（"打分本"有多厚）
  · 训练耗时
  · 作黑打随机的胜率
  · "没学过的局面"占比 —— AI 实战中遇到、却从没在训练里见过的局面比例
    （这一项越高，说明表格越罩不住整个状态空间）
"""

import time

from gomoku_env import GomokuEnv
from agents import QAgent, RandomAgent
from train import train_selfplay

CONFIGS = [(3, 3), (4, 4), (5, 4)]   # (棋盘边长, 连珠数)
GAMES = 40000                         # 每种棋盘的训练局数（固定预算，公平对比）
EVAL = 1000                           # 评估局数

print(f"{'棋盘':>7} {'连珠':>4} {'Q表条目':>12} {'训练耗时':>9} {'作黑胜率':>8} {'没学过的局面占比':>16}")
print("-" * 64)

for bs, wl in CONFIGS:
    agent = QAgent(epsilon=0.3, alpha=0.2, gamma=0.95, seed=0)

    t0 = time.time()
    train_selfplay(agent, GAMES, bs, wl, eps_start=0.30, eps_end=0.05)
    dt = time.time() - t0

    table_size = len(agent.Q)
    trained_keys = {k for (k, a) in agent.Q}      # 训练中见过的所有局面（冻结快照）

    # 评估：纯利用(ε=0)打随机，同时统计"没学过的局面"
    agent.epsilon = 0.0
    rand = RandomAgent(seed=7)
    env = GomokuEnv(bs, wl)
    wins = seen = unseen = 0
    for _ in range(EVAL):
        s = env.reset()
        done = False
        while not done:
            if env.current_player == env.BLACK:
                if agent._state_key(s) in trained_keys:
                    seen += 1
                else:
                    unseen += 1
                a = agent.select_action(s, env.legal_actions())
            else:
                a = rand.select_action(s, env.legal_actions())
            s, _, done, _ = env.step(a)
        if env.winner == env.BLACK:
            wins += 1

    label = f"{bs}x{bs}"
    print(f"{label:>7} {wl:>4} {table_size:>12,} {dt:>8.1f}s "
          f"{wins / EVAL:>8.1%} {unseen / (seen + unseen):>16.1%}")
