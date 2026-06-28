"""
阶段7（6×6 + CNN 化）第一步 —— 最小实验：摸清 nn.Conv2d 的输出形状 & padding
还不搭网络、不训练，只回答一个问题：
    「一张棋盘喂进一个卷积层，吐出来是什么形状？」

对照 aznet.py 现在的 MLP：forward 第一句就 x.flatten() 把棋盘拍扁成一长串；
卷积正相反 —— 全程保持「图」的形状 (batch, 通道, 高, 宽)，不拍扁。
"""

import torch
import torch.nn as nn

# 假装一个 6×6 棋盘的编码：batch=1、2 个通道(我方/对方)、高 6、宽 6。
# 真实场景里它来自 qnet.encode_board(board, player).unsqueeze(0)；
# 这里只看形状，内容填 0 就行（输出形状跟内容无关）。
x = torch.zeros(1, 2, 6, 6)
print("输入形状 (batch, 通道, 高, 宽):", tuple(x.shape))

# —— A：3×3 核，不做 padding（padding 默认 = 0）——
# 核中心只能落在内部 4×4，最外一圈落不进去 → 空间 6 缩成 4。（= 你预测的 4×4）
conv_a = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3)
print("A) 无 padding :", tuple(conv_a(x).shape), " ← 6×6 缩成 4×4，边缘那圈被吃掉")

# —— B：3×3 核 + padding=1（四周补一圈 0）——
# 补了一圈后，核中心也能落到原来的边缘格 → 空间尺寸保持 6×6。这叫 "same" padding。
conv_b = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, padding=1)
print("B) padding=1  :", tuple(conv_b(x).shape), " ← 空间保持 6×6，只是通道从 2 变 32")

# 顺手印证「权重个数与棋盘大小无关」：
# weight 形状 = (out_channels, in_channels, kH, kW)，跟 H/W(棋盘多大) 没有任何关系。
print("\nB 层 weight 形状 (out,in,kH,kW):", tuple(conv_b.weight.shape),
      "→ 权重数 =", conv_b.weight.numel(),
      "(= 32×2×3×3；换 15×15 棋盘这个数一模一样)")
