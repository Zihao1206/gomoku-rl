"""
阶段6 (AlphaZero) 第一步：策略 / 价值网络 —— 双头网络 (two-head net)

对照阶段5 的 QNetwork（见 qnet.py）：
    QNetwork：单头，每个格子输出一个 Q 分（可正可负、互不归一、也没有"整局赢面"这个量）。
    本网络：一段【共享躯干】 + 两个【头】，从同一套"对局面的理解"分出两种输出——
        · policy（策略）头：每格一个分 → 经 softmax 变成"该走这格的先验概率"，全部加起来 = 1。
        · value （价值）头：一个数   → 经 tanh   压到 [-1, +1]，表示"轮到的这位玩家赢面多大"。

输入编码完全复用 qnet.py 的 encode_board：(2, H, W)，且【按"轮到谁"的视角】（图层0=我方子、图层1=对方子）。
所以 value 的语义也跟着是"轮到的这位玩家"的赢面：+1≈我稳赢、-1≈我要输、0≈难分胜负。
"""

import torch
import torch.nn as nn


class PolicyValueNet(nn.Module):
    def __init__(self, board_size, hidden=128):
        super().__init__()                        # 固定写法：先初始化父类 nn.Module
        n_cells = board_size * board_size
        self.board_size = board_size
        self.n_cells = n_cells

        # —— 共享躯干 shared body ——
        # 把编码后的棋盘压成一段 hidden 维的"特征向量"（= 网络对当前局面的理解）。
        # 两个头都从这同一段特征出发：判断"哪步好"和"谁赢面大"本就依赖同一套对局面的认识，
        # 共享躯干 → 省一半参数，且两个任务的梯度一起打磨这套特征，互相促进（正迁移）。
        self.body = nn.Sequential(
            nn.Linear(2 * n_cells, hidden),       # 2 张图(我方/对方)展平 → hidden
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # —— policy 头：hidden → 每格一个"原始分数"(logit) ——
        # 注意：这里【不】在网络里加 softmax，只吐原始分数 logits。
        # 跟做视觉分类一个套路：softmax 留到外面按需做
        # （训练时损失函数用 log_softmax 数值更稳；推断 / 喂给 MCTS 时再 F.softmax 变概率）。
        self.policy_head = nn.Linear(hidden, n_cells)

        # —— value 头：hidden → 1 个数，并在 forward 里过 tanh 压到 [-1, +1] ——
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):
        """
        x 形状 (batch, 2, H, W)。
        返回：
            policy_logits  (batch, n_cells)  每格一个原始分数（还没 softmax）
            value          (batch, 1)        每个样本一个赢面分，已 tanh 到 [-1, 1]
        """
        x = x.flatten(start_dim=1)                # (batch, 2, H, W) → (batch, 2*H*W)
        feat = self.body(x)                       # → (batch, hidden) 共享特征

        policy_logits = self.policy_head(feat)    # → (batch, n_cells)
        value = torch.tanh(self.value_head(feat)) # → (batch, 1)，tanh 压进 [-1, 1]
        return policy_logits, value


if __name__ == "__main__":
    import torch.nn.functional as F
    from gomoku_env import GomokuEnv
    from qnet import encode_board

    # 造一个 3×3 局面：黑下中心、白下角（和 qnet.py 自测同款，方便对照）
    env = GomokuEnv(board_size=3, win_length=3)
    env.reset()
    env.step((1, 1))     # 黑下中心
    env.step((0, 0))     # 白下角
    board = env._get_state()
    env.render()

    # 编码成 (2,H,W)，从"当前轮到的玩家"视角，再加 batch 维 → (1, 2, 3, 3)
    x = encode_board(board, env.current_player).unsqueeze(0)

    net = PolicyValueNet(board_size=3)
    net.eval()                                    # 推断模式（此刻网络没训练，输出是乱的）
    with torch.no_grad():
        policy_logits, value = net(x)

    print("\npolicy logits（原始分数，未归一，9 个数）:")
    print(policy_logits)

    probs = F.softmax(policy_logits, dim=1)       # 在外面 softmax → 变成概率分布
    print("\npolicy 概率（softmax 后）:")
    print(probs)
    print("概率之和（应 ≈ 1.0）:", probs.sum().item())

    print("\nvalue（tanh 后，应落在 -1 ~ 1）:", value.item())
