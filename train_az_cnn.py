"""
阶段7：在 6×6 (win4) 上用【CNN 双头网络】跑 AlphaZero 自举训练。

复用 train_az.train（已支持注入 net）+ eval_mcts.run（已支持传棋盘大小）。
用法：
    python train_az_cnn.py [iterations] [eval_games]
    · iterations  自举训练轮数（默认 20）
    · eval_games  训练后打随机的局数（默认 50；填 0 跳过评估，用于掐表测训练耗时）
模型存到 az_cnn_6x6.pt（state_dict）。设备 cpu（小盘 + 单样本前向，cpu 通常比 mps 快）。
"""

import sys
import torch

from aznet_cnn import PolicyValueNetCNN
from train_az import train
from eval_mcts import run

BS, WL = 6, 4
GAMES_PER_ITER = 20          # 6×6 起步配方（见上一步讨论）
N_SIM = 200
EPOCHS_PER_ITER = 10
LR = 1e-3
DEVICE = "cpu"

iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 20
eval_games = int(sys.argv[2]) if len(sys.argv) > 2 else 50

print("=== AlphaZero+CNN 训练 (6×6, win4) | iters=%d  games/iter=%d  n_sim=%d ==="
      % (iterations, GAMES_PER_ITER, N_SIM))

net = PolicyValueNetCNN(BS)
net = train(BS, WL, iterations=iterations, games_per_iter=GAMES_PER_ITER,
            n_simulations=N_SIM, c_puct=1.0, epochs_per_iter=EPOCHS_PER_ITER,
            lr=LR, seed=0, device=DEVICE, net=net, augment=True)

torch.save(net.state_dict(), "az_cnn_6x6.pt")
print("\n模型已存 az_cnn_6x6.pt")

if eval_games > 0:
    net.eval()
    print("\n=== 训练后：网络版 MCTS vs 随机，各 %d 局（n_sim=%d）===" % (eval_games, N_SIM))
    t = run(eval_games, mcts_is_black=True, n_sim=N_SIM, net=net, board_size=BS, win_length=WL)
    print("执黑：胜 %d (%.0f%%) / 负 %d / 平 %d"
          % (t[1], 100 * t[1] / eval_games, t[2], t[None]))
    t = run(eval_games, mcts_is_black=False, n_sim=N_SIM, net=net, board_size=BS, win_length=WL)
    print("执白：胜 %d (%.0f%%) / 负 %d / 平 %d"
          % (t[2], 100 * t[2] / eval_games, t[1], t[None]))
