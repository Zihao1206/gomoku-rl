"""【B2 冒烟测试】验证 self_play_game 的温度调度落子。

跑一局 3×3 自对弈（未训练网络即可，这里只看"落子调度"对不对，不关心棋力），
打开 verbose 打印每一手走了哪支：
  · 前 temp_moves 手 [采样]：实际落子【不一定】是访问最大的那个（说明在探索）。
  · 第 temp_moves+1 手起 [argmax]：实际落子【必然】等于访问最大的那个（说明在走最优）。
跑：/opt/homebrew/Caskroom/miniforge/base/envs/gomoku/bin/python smoke_temp.py
"""
import random

from aznet import PolicyValueNet
from train_az import self_play_game

net = PolicyValueNet(3)          # 3×3 MLP，未训练——只验证落子逻辑
net.eval()
rng = random.Random(0)

print("=== temp_moves=4：前 4 手应是[采样]、第 5 手起应是[argmax] ===")
self_play_game(net, board_size=3, win_length=3, n_simulations=50,
               c_puct=1.0, rng=rng, temp_moves=4, verbose=True)

print("\n=== 对照 temp_moves=0：整局都应是[argmax]（你预测的'全利用、锁死偏见'）===")
self_play_game(net, board_size=3, win_length=3, n_simulations=50,
               c_puct=1.0, rng=rng, temp_moves=0, verbose=True)
