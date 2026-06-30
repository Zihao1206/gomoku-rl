"""
阶段8（放大到 15×15 标准五子棋 win5）：把网络躯干从「浅 CNN 直接堆」升级成
【残差网络 ResNet】——靠「残差跳连 (skip connection) out = F(x) + x」把网络堆深，
感受野才能盖住 15×15（7 层 3×3 卷积感受野才 15×15），同时不撞梯度消失/退化。

本文件【第一小步】：只实现一个【残差块 ResBlock】并冒烟验证它「形状不变、+x 合法」。
下一步再用它拼出完整的 PolicyValueNetResNet（stem + N×ResBlock + policy/value 双头）。

对照 aznet_cnn.py（阶段7 的浅 CNN，保持原样不动）：那里 body 是三层 Conv 直接串、无跳连。
"""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """
    一个残差块 = 两层 3×3 卷积（各跟一个 BatchNorm）+ 一条把输入原样加回来的「近路」。
    标准 AlphaZero 残差块结构：

        identity = x
        out = relu(bn1(conv1(x)))   # 卷积 → 批归一化 → 激活
        out = bn2(conv2(out))       # 卷积 → 批归一化（先不激活）
        out = out + identity        # ★残差跳连：把原始输入加回来（你学的那个 +x）
        out = relu(out)             # 加完再激活

    BatchNorm（批归一化）：每层卷积后把这批样本在每个通道上拉成均值≈0/标准差≈1，
        稳住「下一层入口的数值分布」，治内部协变量偏移（地基不晃，上层好学、训得快又稳）；
        紧跟可学习的 γ(缩放)/β(平移)，网络要的话能把范围学回来，不损失表达力。
        ⚠️ BN 训练/推断行为不同 → 用前务必 net.train() / net.eval()。

    为什么形状一定对得上、+x 才合法？
        · 两层都是 Conv2d(channels→channels, kernel=3, padding=1)：通道进出相等、H/W 不变；
          BatchNorm2d(channels) 也不改形状。
        · 所以 out 和 identity 形状完全一致 (B, channels, H, W) → 逐元素相加合法。
        这就是为什么残差块内部【通道数保持不变】——要给 +x 留一条形状匹配的近路。
    """

    def __init__(self, channels):
        super().__init__()                                                  # 先初始化父类 nn.Module
        # 两层 3×3 卷积，通道进出都 = channels，padding=1 保持空间尺寸；各跟一个 BatchNorm
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x                              # 留住原始输入——那条「不衰减的近路」
        out = torch.relu(self.bn1(self.conv1(x))) # conv1 → BN → ReLU
        out = self.bn2(self.conv2(out))           # conv2 → BN（先不激活，等加完 +x 再激活）
        out = out + identity                      # ★ F(x) + x：残差跳连
        out = torch.relu(out)                     # 相加之后再过 ReLU
        return out


class PolicyValueNetResNet(nn.Module):
    """
    阶段8 的双头策略价值网（残差版）。整体结构：

        x (B,2,H,W)
          │
        [stem]  Conv2d(2→C,3×3,pad1) + ReLU      ← 先把 2 张输入图升到 C 通道
          │
        [ResBlock] × num_blocks                  ← 残差块堆叠（躯干主体，靠 +x 堆深）
          │  → 共享特征 feat (B,C,H,W)
          ├── policy 头：Conv2d(C→1,1×1) → flatten → (B, n_cells)  每格一个 logit
          └── value  头：Conv2d(C→1,1×1) → flatten → Linear → ReLU → Linear → (B,1) → tanh

    接口与 aznet_cnn.py / aznet.py 完全一致：forward 返回 (policy_logits[B,n_cells], value[B,1])，
    所以 mcts.py / train_az.py 零改即可注入本网络。

    关键：躯干（stem + ResBlock 堆）里【没有一个数依赖 board_size】——卷积只看局部 3×3。
    依赖 board_size 的只有 value 头那个 Linear(n_cells→·)（它要把整盘汇成 1 个数）。
    """

    def __init__(self, board_size, channels=64, num_blocks=5, value_hidden=64):
        super().__init__()
        self.board_size = board_size
        self.n_cells = board_size * board_size
        self.channels = channels
        self.num_blocks = num_blocks

        # —— stem：把输入的 2 张图（我方/对方）升到 C 通道，作为残差块堆的入口 ——
        # 单独一层（不是残差块）：因为残差块要求进出通道相等，而这里要 2→C，通道在变。
        self.stem = nn.Sequential(
            nn.Conv2d(2, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

        # —— 残差躯干：堆 num_blocks 个 ResBlock，每个内部通道都恒定 = channels ——
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(num_blocks)])

        # —— policy 头（沿用 aznet_cnn.py）：1×1 卷积逐格压成 1 个 logit ——
        self.policy_head = nn.Conv2d(channels, 1, kernel_size=1)

        # —— value 头（沿用 aznet_cnn.py）：1×1 卷积 C→1 → 摊平 → 跨格 Linear → 1 个数 ——
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),    # (B,C,H,W) → (B,1,H,W)
            nn.Flatten(),                             # → (B, n_cells)
            nn.Linear(self.n_cells, value_hidden),    # 跨格混合：全局汇总
            nn.ReLU(),
            nn.Linear(value_hidden, 1),               # → 1 个数（tanh 放 forward）
        )

    def forward(self, x):
        """
        x 形状 (batch, 2, H, W)。返回：
            policy_logits (batch, n_cells)  每格一个原始分（未 softmax）
            value         (batch, 1)        赢面分，已 tanh 到 [-1, 1]
        """
        feat = self.stem(x)                                          # (B,2,H,W) → (B,C,H,W)
        feat = self.blocks(feat)                                     # 残差躯干，形状不变 (B,C,H,W)
        policy_logits = self.policy_head(feat).flatten(start_dim=1)  # (B,1,H,W) → (B, n_cells)
        value = torch.tanh(self.value_head(feat))                   # (B,1)，tanh 到 [-1,1]
        return policy_logits, value


if __name__ == "__main__":
    import torch.nn.functional as F

    # ========== 测试 1：单个 ResBlock「形状不变 + +x 生效」 ==========
    torch.manual_seed(0)
    channels = 32
    block = ResBlock(channels)

    # 造一个 15×15、32 通道、batch=4 的假特征图（模拟躯干中间某层的输出）
    x = torch.randn(4, channels, 15, 15)
    out = block(x)

    print("=== 测试 1：单个 ResBlock ===")
    print("输入形状 :", tuple(x.shape))
    print("输出形状 :", tuple(out.shape), " ← 应与输入完全一致（残差块通道/尺寸都不变）")
    assert out.shape == x.shape, "形状变了！+x 就不合法了"

    # 验证「+x 这条近路确实在起作用」：把卷积权重清零（F(x)≡0）后，out 应 = relu(x)
    with torch.no_grad():
        for p in block.parameters():
            p.zero_()
    out_zero = block(x)
    print("权重清零后（F(x)≡0），out == relu(x)：",
          torch.allclose(out_zero, torch.relu(x)),
          " ← 残差块最差也把输入原样传下去（治退化）")

    # ========== 测试 2：完整 PolicyValueNetResNet 喂真 15×15 局面 ==========
    from gomoku_env import GomokuEnv
    from qnet import encode_board

    print("\n=== 测试 2：完整残差网（15×15, win5）===")
    env = GomokuEnv(board_size=15, win_length=5)
    env.reset()
    for mv in [(7, 7), (7, 8), (8, 8), (6, 6)]:   # 黑白随便走几步
        env.step(mv)
    board = env._get_state()

    # 编码 (2,15,15) → 加 batch 维 → (1,2,15,15)，从「当前轮到的玩家」视角
    x = encode_board(board, env.current_player).unsqueeze(0)

    net = PolicyValueNetResNet(board_size=15, channels=64, num_blocks=5)
    n_params = sum(p.numel() for p in net.parameters())
    net.eval()
    with torch.no_grad():
        policy_logits, value = net(x)

    print(f"网络规模 : channels=64, num_blocks=5, 参数量 ≈ {n_params:,}")
    print("policy_logits 形状:", tuple(policy_logits.shape), " ← (1, 225) 每格一个 logit")
    probs = F.softmax(policy_logits, dim=1)
    print("softmax 后概率之和 (应≈1):", round(probs.sum().item(), 4))
    print("value 形状:", tuple(value.shape), " value =", round(value.item(), 4),
          " ← 1 个数，已 tanh 到 [-1,1]")
    print("\n两个头接口与 aznet_cnn.py 一致 → 可直接替换塞进 mcts.py / train_az.py")
