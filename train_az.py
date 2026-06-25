"""
阶段6 (AlphaZero) 第⑦块续：自对弈生成训练数据。

用【网络版 MCTS】自我对弈，每步记录 (局面 state, MCTS 访问分布 π)，
整局下完后按最终胜负给每个局面贴上 value 标签 z（站各局面落子方视角）。
这些 (state, π, z) 就是训练双头网络的监督数据：
    · policy 头学着去模仿 π（搜索提炼出的、比直觉更聪明的落子分布）
    · value  头学着去预测 z（这局最终的真实胜负）
"""

import random

import numpy as np
import torch
import torch.nn.functional as F

from gomoku_env import GomokuEnv
from qnet import encode_board
from aznet import PolicyValueNet
from mcts import mcts_search, _infer_player


def self_play_game(net, board_size, win_length, n_simulations, c_puct, rng, device="cpu"):
    """
    自我对弈一整局，返回训练样本列表 [(state, pi, z), ...]：
        state : 该步的棋盘（numpy 副本）
        pi    : MCTS 访问分布，长度 n_cells 的概率向量（N 归一化；非法/没访问的格 = 0）
        z     : 这局最终结果，站【该局面落子方】视角：赢 +1 / 输 -1 / 平 0
    落子按访问次数【正比采样】（不取 N 最大），制造棋谱多样性——和 ε-贪婪一个道理。
    """
    env = GomokuEnv(board_size, win_length)
    state = env.reset()
    records = []                                       # (state, pi, 该步落子方)
    while not env.done:
        # 跑一次网络版 MCTS，从根拿访问分布
        _, root = mcts_search(state, n_simulations, board_size, win_length,
                              c_puct, rng, net=net, device=device)
        pi = np.zeros(board_size * board_size, dtype=np.float32)
        for (r, c), ch in root.children.items():
            pi[r * board_size + c] = ch.N
        pi /= pi.sum()                                 # 归一化成概率分布
        records.append((state.copy(), pi, env.current_player))

        # 按 N 正比采样落子（探索）
        actions = list(root.children.keys())
        weights = [root.children[a].N for a in actions]
        action = rng.choices(actions, weights=weights)[0]
        state, _, _, _ = env.step(action)

    # 整局结束，按最终胜负给每条样本贴 z（站各局面落子方视角）
    winner = env.winner
    examples = []
    for s, pi, player in records:
        z = 0.0 if winner is None else (1.0 if player == winner else -1.0)
        examples.append((s, pi, z))
    return examples


def examples_to_tensors(examples, device="cpu"):
    """把 [(state, π, z)] 转成训练张量：X (B,2,H,W) / PI (B,n_cells) / Z (B,)。
       每个 state 按"轮到谁"视角 encode（和 value 标签的视角约定一致）。"""
    xs = [encode_board(s, _infer_player(s)) for (s, _, _) in examples]
    X = torch.stack(xs).to(device)
    PI = torch.tensor(np.array([p for (_, p, _) in examples]),
                      dtype=torch.float32, device=device)
    Z = torch.tensor([z for (_, _, z) in examples],
                     dtype=torch.float32, device=device)
    return X, PI, Z


def compute_loss(net, X, PI, Z):
    """
    AlphaZero 的双头损失：
        policy_loss = 软标签交叉熵 = -Σ_a π(a)·log p(a)   （p = softmax(logits)）
        value_loss  = MSE(value, z)
        total       = policy_loss + value_loss
    policy 用 log_softmax（数值稳定）——正好接上 aznet "网络只吐 logits、softmax 留外面"的设计。
    """
    logits, value = net(X)                          # (B, n_cells), (B, 1)
    logp = F.log_softmax(logits, dim=1)             # log p(a)
    policy_loss = -(PI * logp).sum(dim=1).mean()    # 每样本对动作求和，再对 batch 平均
    value_loss = F.mse_loss(value.squeeze(1), Z)    # (value - z)²
    return policy_loss + value_loss, policy_loss, value_loss


def train(board_size, win_length, iterations, games_per_iter, n_simulations,
          c_puct, epochs_per_iter, lr, seed, device="cpu"):
    """
    AlphaZero 自举训练循环：反复 {用当前网络自对弈攒数据 → 拿数据训练网络}。
    每轮数据都来自【上一轮训练后】的网络——网络越强、棋谱越好、训练目标越好……滚雪球。
    返回训练好的网络。
    """
    net = PolicyValueNet(board_size).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    rng = random.Random(seed)
    for it in range(iterations):
        # 1) 自对弈收集数据（用当前网络）
        net.eval()
        data = []
        for _ in range(games_per_iter):
            data += self_play_game(net, board_size, win_length,
                                   n_simulations, c_puct, rng, device)
        # 2) 拿这批数据做若干步梯度下降
        net.train()
        X, PI, Z = examples_to_tensors(data, device)
        for _ in range(epochs_per_iter):
            optimizer.zero_grad()
            total, pl, vl = compute_loss(net, X, PI, Z)
            total.backward()
            optimizer.step()
        print("iter %2d | 样本 %3d | policy=%.3f  value=%.3f  total=%.3f"
              % (it, len(data), pl.item(), vl.item(), total.item()))
    return net


if __name__ == "__main__":
    from eval_mcts import run

    BS, WL = 3, 3
    print("=== AlphaZero 自举训练（3×3）===")
    net = train(BS, WL, iterations=15, games_per_iter=10, n_simulations=100,
                c_puct=1.0, epochs_per_iter=10, lr=1e-3, seed=0)
    net.eval()

    print("\n=== 训练后验证：网络版 MCTS vs 随机，各 50 局 ===")
    print("（对比基线：未训练网络版 执黑 96%、纯 rollout 执黑 100%）")
    t = run(50, mcts_is_black=True, net=net)
    print("训练后网络 执黑：胜 %d (%.0f%%) / 负 %d / 平 %d"
          % (t[1], 100 * t[1] / 50, t[2], t[None]))
    t = run(50, mcts_is_black=False, net=net)
    print("训练后网络 执白：胜 %d (%.0f%%) / 负 %d / 平 %d"
          % (t[2], 100 * t[2] / 50, t[1], t[None]))
