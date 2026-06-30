"""
阶段6 (AlphaZero) 第⑦块续：自对弈生成训练数据。

用【网络版 MCTS】自我对弈，每步记录 (局面 state, MCTS 访问分布 π)，
整局下完后按最终胜负给每个局面贴上 value 标签 z（站各局面落子方视角）。
这些 (state, π, z) 就是训练双头网络的监督数据：
    · policy 头学着去模仿 π（搜索提炼出的、比直觉更聪明的落子分布）
    · value  头学着去预测 z（这局最终的真实胜负）
"""

import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from gomoku_env import GomokuEnv
from qnet import encode_board
from aznet import PolicyValueNet
from mcts import mcts_search, _infer_player
from augment import augment_example


def self_play_game(net, board_size, win_length, n_simulations, c_puct, rng,
                   device="cpu", temp_moves=4, verbose=False,
                   dir_eps=0.25, dir_alpha=0.3):
    """
    自我对弈一整局，返回训练样本列表 [(state, pi, z), ...]：
        state : 该步的棋盘（numpy 副本）
        pi    : MCTS 访问分布，长度 n_cells 的概率向量（N 归一化；非法/没访问的格 = 0）
        z     : 这局最终结果，站【该局面落子方】视角：赢 +1 / 输 -1 / 平 0
    【B2 温度调度】落子分两段：前 temp_moves 手按访问数【正比采样】(高温/探索，
    制造开局多样性)；第 temp_moves+1 手起改走访问数最大的那手 (argmax/低温/走最优，
    把残局走干净，以免随机走输污染整局的 z 标签)。
    注意：存进 records 的 pi 始终是 τ=1 的访问分布(软标签)，【不随温度变】——
    温度只决定"实际落哪一步"，不动 policy 头的训练目标。
    """
    env = GomokuEnv(board_size, win_length)
    state = env.reset()
    records = []                                       # (state, pi, 该步落子方)
    move_count = 0                                     # 这局下到第几手（从 0 起，循环外初始化）
    while not env.done:
        # 跑一次网络版 MCTS，从根拿访问分布
        _, root = mcts_search(state, n_simulations, board_size, win_length,
                              c_puct, rng, net=net, device=device,
                              dir_eps=dir_eps, dir_alpha=dir_alpha)
        pi = np.zeros(board_size * board_size, dtype=np.float32)
        for (r, c), ch in root.children.items():
            pi[r * board_size + c] = ch.N
        pi /= pi.sum()                                 # 归一化成概率分布（软标签，不随温度变）
        records.append((state.copy(), pi, env.current_player))

        # 【B2 温度调度】前 temp_moves 手高温采样、之后 argmax
        actions = list(root.children.keys())
        weights = [root.children[a].N for a in actions]
        if move_count < temp_moves:
            action = rng.choices(actions, weights=weights)[0]   # ∝N 采样（高温/探索）
        else:
            best_i = weights.index(max(weights))                # 访问数最大者的下标
            action = actions[best_i]                            # 取那一手（argmax/低温）
        if verbose:                                             # 调试观察：这一手走了哪支
            best_a = actions[weights.index(max(weights))]
            branch = "采样  " if move_count < temp_moves else "argmax"
            print("  手%2d [%s] 落子=%s  访问最大=%s  weights=%s"
                  % (move_count + 1, branch, action, best_a,
                     [int(w) for w in weights]))
        state, _, _, _ = env.step(action)
        move_count += 1                                # 走完一手，计数 +1

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
          c_puct, epochs_per_iter, lr, seed, device="cpu", net=None, augment=False,
          temp_moves=4, dir_eps=0.25, dir_alpha=0.3,
          use_buffer=False, buffer_capacity=20000, batch_size=256, train_steps=30,
          checkpoint_path=None, checkpoint_every=0, resume=False):
    """
    AlphaZero 自举训练循环：反复 {用当前网络自对弈攒数据 → 拿数据训练网络}。
    每轮数据都来自【上一轮训练后】的网络——网络越强、棋谱越好、训练目标越好……滚雪球。
    返回训练好的网络。

    【阶段8 checkpoint 断点续训】（默认全关，不影响 3×3/6×6 旧调用）：
        checkpoint_path  : 存盘路径（None=不存）
        checkpoint_every : 每隔几轮存一次（0=不存）；最后一轮总会存
        resume           : True 且 checkpoint_path 存在时，从断点恢复
                           网络权重 + 优化器状态(Adam动量) + 下一轮序号 + 回放池 + rng
    """
    if net is None:                                 # 默认 MLP（向后兼容 3×3 老实验）
        net = PolicyValueNet(board_size)
    net = net.to(device)                            # 传进来的 CNN / MLP 都走这里上设备
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    rng = random.Random(seed)
    buffer = deque(maxlen=buffer_capacity)          # 【B4】回放池：use_buffer=True 时启用

    # —— 断点续训：把上次存的四样东西原样恢复，做到"无缝接上" ——
    start_iter = 0
    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        # weights_only=False：我们的 ckpt 里有 numpy/rng 等非张量对象，需完整反序列化（自存自读、可信）
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["net"])
        optimizer.load_state_dict(ckpt["optimizer"])   # 恢复 Adam 一阶/二阶动量，避免退化成裸 SGD
        buffer.extend(ckpt["buffer"])                  # 恢复回放池（含稀有防守样本）
        rng.setstate(ckpt["rng"])                      # 恢复随机数状态，序列可复现
        start_iter = ckpt["iter"]                      # 从"下一轮"接着跑
        print("[续训] 从 %s 恢复：下一轮=%d，池=%d" % (checkpoint_path, start_iter, len(buffer)))

    def _save_ckpt(next_iter):
        torch.save({
            "iter": next_iter,                         # 下次该从第几轮开始
            "net": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "buffer": list(buffer),
            "rng": rng.getstate(),
        }, checkpoint_path)

    for it in range(start_iter, iterations):
        # 1) 自对弈收集数据（用当前网络）
        net.eval()
        data = []
        for _ in range(games_per_iter):
            data += self_play_game(net, board_size, win_length,
                                   n_simulations, c_puct, rng, device,
                                   temp_moves=temp_moves,
                                   dir_eps=dir_eps, dir_alpha=dir_alpha)
        # 1.5) 对称性数据增强：每条样本 → 8 条等价（D4 全对称），修朝向盲区
        if augment:
            aug = []
            for s, pi, z in data:
                aug += augment_example(s, pi, z, board_size)
            data = aug
        # 2) 训练：B4 用回放池抽 minibatch；否则（旧逻辑）整批做 epochs 次全量梯度
        net.train()
        if use_buffer:
            buffer.extend(data)                     # 【B4】新数据进池（不直接训这批、也不丢）
            pool = list(buffer)                     # 转一次 list，下面多次抽样复用
            for _ in range(train_steps):            # 每轮抽 train_steps 个 minibatch 训
                batch = rng.sample(pool, min(batch_size, len(pool)))   # 随机抽→打破相关性
                X, PI, Z = examples_to_tensors(batch, device)
                optimizer.zero_grad()
                total, pl, vl = compute_loss(net, X, PI, Z)
                total.backward()
                optimizer.step()
            print("iter %2d | 池 %5d | policy=%.3f  value=%.3f  total=%.3f"
                  % (it, len(buffer), pl.item(), vl.item(), total.item()))
        else:
            X, PI, Z = examples_to_tensors(data, device)
            for _ in range(epochs_per_iter):
                optimizer.zero_grad()
                total, pl, vl = compute_loss(net, X, PI, Z)
                total.backward()
                optimizer.step()
            print("iter %2d | 样本 %3d | policy=%.3f  value=%.3f  total=%.3f"
                  % (it, len(data), pl.item(), vl.item(), total.item()))

        # —— 存 checkpoint：每 checkpoint_every 轮存一次，最后一轮也总存 ——
        if checkpoint_path and checkpoint_every and \
           ((it + 1) % checkpoint_every == 0 or it + 1 == iterations):
            _save_ckpt(it + 1)
            print("   ↳ checkpoint 已存 %s (iter=%d)" % (checkpoint_path, it + 1))
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
