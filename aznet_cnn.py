"""
阶段7（6×6 + CNN 化）：把 PolicyValueNet 的 MLP 躯干换成【卷积躯干】(CNN body)。

对照 aznet.py（MLP 版）：
    · forward 第一句 x.flatten() 把棋盘拍扁成一长串，丢掉了棋盘几何；
    · 第一层 nn.Linear(2*n_cells, hidden) 的输入维度【写死了棋盘大小】。
本版（CNN）：
    · 躯干全是 Conv2d(kernel=3, padding=1)，全程保持 (batch, 通道, H, W)，不拍扁、不缩；
    · ——注意—— 躯干代码里【压根没出现 board_size】：同一套躯干 6×6 / 15×15 通用。

两个头的"分水岭"：
    · policy 头：逐格、与盘大小无关 → 用 1×1 卷积（保住 6×6 空间，每格一个 logit）。
    · value  头：全局、要把整盘汇成 1 个数 → 必须【跨格】，用 flatten + Linear。

接口与 aznet.py 完全一致：forward 返回 (policy_logits[B,n_cells], value[B,1])，
所以可直接替换 MLP 版塞进 mcts.py / train_az.py。
"""

import torch
import torch.nn as nn


class PolicyValueNetCNN(nn.Module):
    def __init__(self, board_size, channels=32, value_hidden=64):
        super().__init__()                       # 固定写法：先初始化父类 nn.Module
        self.board_size = board_size
        self.n_cells = board_size * board_size
        self.channels = channels

        # —— 卷积躯干 conv body ——
        # 三层 3×3 卷积，每层都 padding=1 → 空间尺寸一路保持（不缩边）。
        # 通道口径对齐：第一层 in=2（encode_board 的 我方/对方 两张图），之后 in=out=channels。
        # 关键：这段【没有一个数依赖 board_size】——卷积只看局部 3×3，与棋盘多大无关。
        self.body = nn.Sequential(
            nn.Conv2d(2, channels, kernel_size=3, padding=1),         # (B,2,H,W) → (B,C,H,W)
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),  # (B,C,H,W) → (B,C,H,W)
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),  # (B,C,H,W) → (B,C,H,W)
            nn.ReLU(),
        )

        # —— policy 头：1×1 卷积，逐格把 C 个特征压成 1 个分 ——
        # 等价于"每个格子上套同一个 Linear(C→1)"；输出 (B,1,H,W)，forward 里展平成 (B,n_cells)。
        # 权重只有 C 个(+1)，与棋盘大小无关；展平顺序=行优先=row*W+col，对上项目动作下标约定。
        self.policy_head = nn.Conv2d(channels, 1, kernel_size=1)       # (B,C,H,W) → (B,1,H,W)

        # —— value 头：把整盘 (B,C,H,W) 收成 1 个赢面分 ——
        # 三步：① 1×1 卷积 C→1（每格 C 特征先压成 1）→ (B,1,H,W)
        #      ② Flatten → (B, n_cells)
        #      ③ Linear 跨格混合 → 收成 1 个数（tanh 放 forward 里）
        # 这里 flatten+Linear 是【该用】的：value 本就是"看遍整盘、认这块盘"的全局总评。
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),    # (B,C,H,W) → (B,1,H,W)：通道 C→1
            nn.Flatten(),                             # (B,1,H,W) → (B, n_cells)：摊平空间
            nn.Linear(self.n_cells, value_hidden),    # 跨格混合：每个输出都连到全部格子
            nn.ReLU(),
            nn.Linear(value_hidden, 1),               # → 1 个数（还没 tanh）
        )

    def forward(self, x):
        """
        x 形状 (batch, 2, H, W)。返回：
            policy_logits (batch, n_cells)  每格一个原始分（未 softmax，留外面做）
            value         (batch, 1)        赢面分，已 tanh 到 [-1, 1]
        """
        feat = self.body(x)                                       # (B, C, H, W) 共享特征
        policy_logits = self.policy_head(feat).flatten(start_dim=1)  # (B,1,H,W) → (B, n_cells)
        value = torch.tanh(self.value_head(feat))                # (B, 1)，tanh 压进 [-1,1]
        return policy_logits, value


if __name__ == "__main__":
    import torch.nn.functional as F
    from gomoku_env import GomokuEnv
    from qnet import encode_board

    # 造一个真实的 6×6 局面（win_length 暂填 4，网络不关心；正式训练时再定）
    env = GomokuEnv(board_size=6, win_length=4)
    env.reset()
    for mv in [(2, 2), (2, 3), (3, 3)]:           # 黑、白、黑 随便走三步
        env.step(mv)
    board = env._get_state()
    env.render()

    # 编码 (2,6,6) → 加 batch 维 → (1,2,6,6)，从"当前轮到的玩家"视角
    x = encode_board(board, env.current_player).unsqueeze(0)

    net = PolicyValueNetCNN(board_size=6, channels=32)
    net.eval()                                    # 推断模式（此刻没训练，输出是乱的）
    with torch.no_grad():
        policy_logits, value = net(x)

    print("\npolicy_logits 形状:", tuple(policy_logits.shape), " ← (1, 36) 每格一个 logit")
    probs = F.softmax(policy_logits, dim=1)       # 外面 softmax → 概率分布
    print("softmax 后概率之和 (应≈1):", round(probs.sum().item(), 4))
    print("value 形状:", tuple(value.shape), " value =", round(value.item(), 4),
          " ← 1 个数，已 tanh 到 [-1,1]")
    print("\n两个头接口与 aznet.py MLP 版一致 → 可直接替换塞进 mcts.py / train_az.py")
