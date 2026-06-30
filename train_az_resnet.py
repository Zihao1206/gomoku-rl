"""
阶段8：在 15×15 (win5) 标准五子棋上，用【残差网络】跑 AlphaZero 自举训练。

复用 train_az.train（已支持注入 net + checkpoint 断点续训）。
对照 train_az_cnn.py（6×6 浅 CNN 入口）：本入口换成 PolicyValueNetResNet、棋盘 15×15、win5，
并开启 checkpoint（长训必备——中断后可续）。

用法：
    python train_az_resnet.py [iterations] [resume]
        · iterations  自举训练轮数（默认 100）
        · resume      传 "resume" 则从 checkpoint 断点续训；不传则从头开始
    例：
        python train_az_resnet.py 100            # 从头训 100 轮
        python train_az_resnet.py 100 resume     # 被中断后，接着训到 100 轮

产出：
    az_resnet_15x15.ckpt  断点（网络+优化器+轮序+回放池+rng），每轮存一次
    az_resnet_15x15.pt    最终模型（只存 state_dict，供对弈/诊断加载）

设备：自动检测——有 CUDA(4070) 用 cuda；否则(本 Mac 无 N 卡)落 cpu。
      不用 mps：实测 AZ 自对弈是 batch=1 逐局面前向，mps 反比 cpu 慢（见 smoke_az_resnet.py）。
      → 同一份代码两台机器都对，无需改设备。
"""

import sys
import torch

from aznet_resnet import PolicyValueNetResNet
from train_az import train

# ===== 棋盘 =====
BS, WL = 15, 5                       # 标准五子棋

# ===== 网络（"尽量强"配方：更宽更深，4070 扛得住）=====
CHANNELS = 128                       # 通道加宽 → 容量更大，学更多棋形特征
NUM_BLOCKS = 7                       # 加深；感受野 stem(1)+7块×2=15 层 → (2·15+1) 远超盘宽

# ===== 训练配方（"尽量强"，主要在 4070/CUDA 上跑）=====
GAMES_PER_ITER = 25                  # 每轮自对弈局数
N_SIM = 400                          # MCTS 每手模拟次数（搜得深，自对弈棋谱质量高）
EPOCHS_PER_ITER = 10                 # 非 buffer 分支才用（这里走 buffer）
LR = 1e-3
TEMP_MOVES = 10                      # 【B2】15×15 开局长，前 10 手高温采样制造多样性
DIR_EPS = 0.25                       # 【B3】根 Dirichlet 噪声占比
DIR_ALPHA = 0.15                     # 【B3】≈10/合法手数；15×15 合法手多 → alpha 调小更集中
USE_BUFFER = True                    # 【B4】回放池 + minibatch
BUFFER_CAP = 200000                  # 【B5】15×15 样本稀 → 池子要大才存得住稀有防守样本
BATCH_SIZE = 512                     # GPU 大 batch 才划算
TRAIN_STEPS = 200                    # 每轮抽多少 minibatch 训

# ===== checkpoint =====
CKPT_PATH = "az_resnet_15x15.ckpt"
CKPT_EVERY = 1                       # 每轮都存——15×15 每轮都很贵，不容丢
MODEL_PATH = "az_resnet_15x15.pt"

# ===== 设备：自动检测，两台机器同一份代码 =====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 100
resume = len(sys.argv) > 2 and sys.argv[2] == "resume"

print("=== AlphaZero+ResNet 训练 (15×15, win5) | %s ===" % DEVICE)
print("iters=%d  games/iter=%d  n_sim=%d  blocks=%d  resume=%s\n"
      % (iterations, GAMES_PER_ITER, N_SIM, NUM_BLOCKS, resume))

net = PolicyValueNetResNet(BS, channels=CHANNELS, num_blocks=NUM_BLOCKS)
net = train(
    BS, WL,
    iterations=iterations, games_per_iter=GAMES_PER_ITER,
    n_simulations=N_SIM, c_puct=1.0, epochs_per_iter=EPOCHS_PER_ITER,
    lr=LR, seed=0, device=DEVICE, net=net, augment=True,
    temp_moves=TEMP_MOVES, dir_eps=DIR_EPS, dir_alpha=DIR_ALPHA,
    use_buffer=USE_BUFFER, buffer_capacity=BUFFER_CAP,
    batch_size=BATCH_SIZE, train_steps=TRAIN_STEPS,
    checkpoint_path=CKPT_PATH, checkpoint_every=CKPT_EVERY, resume=resume,
)

torch.save(net.state_dict(), MODEL_PATH)
print("\n最终模型已存 %s" % MODEL_PATH)
