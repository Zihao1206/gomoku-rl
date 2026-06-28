"""
公平测试：剥离"搜索"、单看"网络"——
让【训练后网络版 MCTS】对打【纯 rollout 版 MCTS】，【同样的 n_sim】，黑白各 N 局。
搜索预算一样，唯一差别 = "用网络的 policy/value 导航" vs "用随机 rollout 估值"。
    · 网络若赢 / 守平 → 它真学到了棋（值回搜索的票）。
    · 网络若输给 rollout → 它现在比"零知识搜索"还弱（欠练的铁证）。

注：网络版 MCTS 是确定性的（argmax 访问数、net 评估不用随机）；rollout 版靠随机摸盘，
    所以多局之间的不同，全由 rollout 的 rng 在 N 局间滚动产生 → 才有胜负分布。

用法：python eval_net_vs_rollout.py [每色局数 N=20] [n_sim=200]
"""

import sys
from collections import Counter

import torch

from gomoku_env import GomokuEnv
from aznet_cnn import PolicyValueNetCNN
from mcts import MCTSAgent
from play import play_game

BS, WL = 6, 4
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20          # 每个颜色对打多少局
N_SIM = int(sys.argv[2]) if len(sys.argv) > 2 else 200     # 两边一致的模拟数
MODEL = "az_cnn_6x6.pt"

net = PolicyValueNetCNN(BS)
net.load_state_dict(torch.load(MODEL, map_location="cpu"))
net.eval()

env = GomokuEnv(BS, WL)


def duel(net_is_black):
    """net 执黑或执白，与 rollout 对打 N 局，返回 (net胜, rollout胜, 平)。"""
    net_mcts = MCTSAgent(BS, WL, n_simulations=N_SIM, net=net, device="cpu", seed=0, name="NET")
    roll_mcts = MCTSAgent(BS, WL, n_simulations=N_SIM, net=None, seed=0, name="ROLLOUT")
    black, white = (net_mcts, roll_mcts) if net_is_black else (roll_mcts, net_mcts)
    net_color = env.BLACK if net_is_black else env.WHITE
    roll_color = env.WHITE if net_is_black else env.BLACK
    tally = Counter()
    for _ in range(N):                                     # 复用 agent：rollout 的 rng 逐局滚动 → 每局不同
        tally[play_game(env, black, white)] += 1
    return tally[net_color], tally[roll_color], tally[None]


print("=== 训练后网络 MCTS  vs  纯 rollout MCTS （6×6 win4, 双方 n_sim=%d, 各 %d 局）===" % (N_SIM, N))
nw, rw, d = duel(net_is_black=True)
print("网络执黑：网络胜 %2d / rollout 胜 %2d / 平 %2d" % (nw, rw, d))
nw, rw, d = duel(net_is_black=False)
print("网络执白：网络胜 %2d / rollout 胜 %2d / 平 %2d" % (nw, rw, d))
