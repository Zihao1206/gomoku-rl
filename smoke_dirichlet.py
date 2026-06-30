"""【B3 冒烟】Dirichlet 根噪声直接作用于 MCTS：对一个必防局面，
对比有/无噪声时防守手的根先验 P 与访问数 N。直接调 mcts_search(dir_eps=...)，
不经训练管线——单看"噪声把被冷落的防守手抬起来"这一机制对不对。
跑：/opt/homebrew/Caskroom/miniforge/base/envs/gomoku/bin/python smoke_dirichlet.py
"""
import random

import numpy as np
import torch

from aznet_cnn import PolicyValueNetCNN
from mcts import mcts_search

BS, WL = 6, 4
net = PolicyValueNetCNN(BS)
net.load_state_dict(torch.load("az_cnn_6x6.pt", map_location="cpu"))
net.eval()

# diag 的"横 H"必防局面：黑三连占第2行 (2,1)(2,2)(2,3)、白堵一头 (2,0)，白必防 (2,4)
b = np.zeros((BS, BS), dtype=np.int64)
for r, c in [(2, 1), (2, 2), (2, 3)]:
    b[r, c] = 1
for r, c in [(2, 0), (5, 5)]:
    b[r, c] = 2
MUST = (2, 4)


def one(dir_eps, seed):
    best, root = mcts_search(b, 400, BS, WL, 1.0, random.Random(seed),
                             net=net, device="cpu", dir_eps=dir_eps, dir_alpha=0.3)
    ch = root.children[MUST]
    hit = "  ✅MCTS 选中防守点!" if best == MUST else ""
    print("    seed%d | 防守手(2,4): 根先验 P=%.3f  访问 N=%3d%s"
          % (seed, ch.P, ch.N, hit))


print("必防局面：白三连占第2行、白必防点=(2,4)。n_sim=400, dir_alpha=0.3\n")
print("【无噪声 dir_eps=0】(网络版 MCTS 确定性，跑 1 次即代表)")
one(0.0, 0)
print("\n【有噪声 dir_eps=0.25】(每个 seed 噪声不同，看 (2,4) 被抬高/被搜到)")
for s in range(6):
    one(0.25, s)
