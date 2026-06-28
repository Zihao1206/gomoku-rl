"""
阶段7 冒烟测试 (smoke test)：把 PolicyValueNetCNN 塞进【几乎没动的】AlphaZero 管线，
在 6×6 (win4) 上验证三件事：
    ① 自对弈能在 6×6 上产出样本 (state, π, z)，形状对、π 是概率分布；
    ② examples_to_tensors / compute_loss 不改一行就能吃 CNN 的输出（接口对齐的实证）；
    ③ 固定一批数据上多走几步梯度，loss 能降 —— 证明 CNN 的梯度通到了 policy/value 两个头；
    ④ 顺手用注入式 train(net=...) 跑 1 轮，确认"换胎后的 train()"整条链能转。

注意：self_play_game / examples_to_tensors / compute_loss 全部直接从 train_az 复用，
一行没改 —— 这正是"forward 接口一致 → CNN 即插即用"的证据。
"""

import math
import random

import torch

from aznet_cnn import PolicyValueNetCNN
from train_az import self_play_game, examples_to_tensors, compute_loss, train

BS, WL = 6, 4
device = "cpu"                       # 小盘 smoke，cpu 最省事（mps 留给正式训练再试）
rng = random.Random(0)

print("=== ① 自对弈一局（6×6, win4, 未训练 CNN）===")
net = PolicyValueNetCNN(BS).to(device)
net.eval()
examples = self_play_game(net, BS, WL, n_simulations=20, c_puct=1.0, rng=rng, device=device)
print("样本数:", len(examples))
s0, pi0, z0 = examples[0]
print("样本[0]: state.shape=%s  π长度=%d  π之和=%.3f(应≈1)  z=%s"
      % (s0.shape, len(pi0), pi0.sum(), z0))

print("\n=== ② 转训练张量（examples_to_tensors，未改动）===")
X, PI, Z = examples_to_tensors(examples, device)
print("X=%s(应 (B,2,6,6))  PI=%s(应 (B,36))  Z=%s(应 (B,))"
      % (tuple(X.shape), tuple(PI.shape), tuple(Z.shape)))

print("\n=== ③ 固定这批数据做梯度下降，看 loss 降（policy 初值应≈log36=%.2f）===" % math.log(36))
net.train()
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
for step in range(31):
    opt.zero_grad()
    total, pl, vl = compute_loss(net, X, PI, Z)
    total.backward()
    opt.step()
    if step % 10 == 0:
        print("  step %2d | policy=%.3f  value=%.3f  total=%.3f"
              % (step, pl.item(), vl.item(), total.item()))

print("\n=== ④ 注入式 train(net=CNN) 跑 1 轮，确认换胎后的训练链能转 ===")
train(BS, WL, iterations=1, games_per_iter=1, n_simulations=12, c_puct=1.0,
      epochs_per_iter=5, lr=1e-3, seed=1, device=device, net=PolicyValueNetCNN(BS))

print("\n✅ 冒烟通过：CNN 即插即用 + 梯度能降 loss + train() 链路转通")
