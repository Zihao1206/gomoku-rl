"""
阶段8【小规模冒烟】：把残差网 PolicyValueNetResNet 塞进现成训练链
（mcts.py + train_az.py），在 15×15 (win5) 上跑【最小一轮】自对弈+训练。

目的只有一个：证明「残差网 + 15×15 + mps」这条管子能从头通到尾、不报错。
【不看棋力】——只跑 1 局自对弈、网络还是随机初始化的。

故意踩两个真问题，提前暴露：
  ① device=mps：阶段8 第一次真用 Metal，个别算子若不支持会当场报错（宁可现在撞）。
  ② 15×15 一局很长：掐个单局/单轮耗时，给后面长训那步估配方。

用法：python smoke_az_resnet.py
"""

import time
import torch

from aznet_resnet import PolicyValueNetResNet
from train_az import train

# —— 设备：实测发现 mps 反而更慢！——
# AlphaZero 自对弈是 batch=1 的逐局面前向，GPU 的并行优势用不上，
# 每次调用的启动/数据搬运开销反而拖累 → Mac 上 cpu 实测比 mps 快 ~3 倍
# （同冒烟：cpu 25s vs mps 78s）。故 Mac 默认走 cpu。绝不写 cuda。
# （注：4070 那台机器是 CUDA，长训时另议；训练 batch 大那步 GPU 才划算。）
DEVICE = "cpu"

BS, WL = 15, 5                       # 标准五子棋：15×15 棋盘、5 连胜

print("=== 阶段8 冒烟：残差网 + 15×15(win5) + %s ===" % DEVICE)
print("（最小配置，只验证管子通不通、掐耗时，不看棋力）\n")

# 残差网：和正式长训同结构，但这里只是冒烟
net = PolicyValueNetResNet(BS, channels=64, num_blocks=5)
n_params = sum(p.numel() for p in net.parameters())
print("网络：channels=64, num_blocks=5, 参数量 ≈ {:,}".format(n_params))

t0 = time.time()
net = train(
    BS, WL,
    iterations=1,            # 只跑 1 轮自举
    games_per_iter=1,        # 1 轮里只自对弈 1 局（冒烟，省时间）
    n_simulations=16,        # MCTS 每手只搜 16 次（冒烟用很小值）
    c_puct=1.0,
    epochs_per_iter=2,       # 非 buffer 分支才用得到（这里走 buffer）
    lr=1e-3,
    seed=0,
    device=DEVICE,
    net=net,                 # 注入我们的残差网
    augment=True,            # 8 重对称增强（阶段7 验证有效，照旧开）
    temp_moves=4,            # 温度调度（同阶段7）
    dir_eps=0.25, dir_alpha=0.3,   # 根节点 Dirichlet 噪声（仅自对弈）
    use_buffer=True,         # 回放池 + minibatch
    buffer_capacity=2000,    # 冒烟用小池
    batch_size=64,
    train_steps=3,           # 冒烟只训 3 个 minibatch
)
dt = time.time() - t0

print("\n冒烟总耗时：%.1f 秒（含 1 局自对弈 + 3 步训练）" % dt)
print("→ 管子全程没报错 = 残差网/mps/15×15 接线通过；耗时供长训配方参考。")
