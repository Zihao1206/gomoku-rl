"""
DQN 第一步 (5.1)：棋盘编码 + 最简单的 Q 网络
先把"大脑结构"搭出来，还不训练。

- encode_board：把棋盘变成神经网络能吃的数字张量（用"图层/通道"表示）
- QNetwork：一个最简单的多层感知机(MLP)，输入棋盘、输出"每个格子的 Q 分"
"""

import numpy as np
import torch
import torch.nn as nn


def encode_board(board, player):
    """
    把棋盘编码成 (2, H, W) 的张量，【从 player 的视角】看：
        图层 0：我方的子（是 1，否 0）
        图层 1：对方的子（是 1，否 0）
    用"我方/对方"而不是"黑/白"，是为了让同一个网络对黑白两边都通用、学得更顺。
    """
    opponent = 3 - player                        # BLACK=1 ↔ WHITE=2，对手就是 3-player
    me = (board == player).astype(np.float32)    # 我方子的位置图
    opp = (board == opponent).astype(np.float32) # 对方子的位置图
    planes = np.stack([me, opp])                 # 叠成 2 张图 → (2, H, W)
    return torch.from_numpy(planes)


class QNetwork(nn.Module):
    """
    最简单的 Q 网络（多层感知机 MLP）：
        输入：编码后的棋盘（2 张图展平成一长串数字）
        输出：每个格子一个 Q 分（共 H×W 个）
    """

    def __init__(self, board_size, hidden=128):
        super().__init__()                       # 固定写法：先初始化父类 nn.Module
        n_cells = board_size * board_size
        self.net = nn.Sequential(                # 一摞按顺序执行的层
            nn.Linear(2 * n_cells, hidden),      # 全连接层：2×格子数 个输入 → hidden 个
            nn.ReLU(),                           # 激活函数：给网络"掰弯"的能力(否则只是直线)
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_cells),          # 输出层：→ 每个格子一个 Q 值
        )

    def forward(self, x):
        """x 形状 (batch, 2, H, W)；把每个样本的 2 张图展平后过网络。"""
        x = x.flatten(start_dim=1)               # → (batch, 2*H*W)
        return self.net(x)                       # → (batch, n_cells)


if __name__ == "__main__":
    from gomoku_env import GomokuEnv

    env = GomokuEnv(board_size=3, win_length=3)
    env.reset()
    env.step((1, 1))     # 黑下中心
    env.step((0, 0))     # 白下角
    board = env._get_state()
    env.render()

    # 1) 编码：从"当前轮到的玩家"视角看棋盘
    x = encode_board(board, env.current_player)
    print("编码张量形状 (2张图, 高, 宽):", tuple(x.shape))
    print(x)

    # 2) 建网络，做一次前向：喂一个棋盘，吐出每个格子的 Q 分
    net = QNetwork(board_size=3)
    q = net(x.unsqueeze(0))          # unsqueeze(0)：加一个 batch 维 → (1, 2, 3, 3)
    print("\n网络输出的 Q 分 (还没训练，所以是乱的):")
    print(q)
    print("输出形状 (batch, 格子数):", tuple(q.shape))
