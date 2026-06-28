"""
人机对战（AlphaZero + CNN 版）—— 跟你训练好的 6×6 CNN 网络下一盘。

AI 的"脑子" = MCTSAgent 包着加载的 PolicyValueNetCNN：
    每步跑一次网络版 MCTS（policy 当先验、value 评估），取访问次数最高的落子。

对照 play_human_dqn.py：结构逐字一样，只把 AI 那一行从 DQNAgent 换成 MCTSAgent(net=...)
—— 又是"所有 agent 共用 select_action 接口"的红利。

用法：
    python play_human_az_cnn.py                       # 默认 6×6(win4)，加载 az_cnn_6x6.pt
    python play_human_az_cnn.py 6 4                    # 显式指定棋盘
    python play_human_az_cnn.py 6 4 az_cnn_6x6.pt 400 # 再指定模型 / 每步模拟数
"""

import os
import sys

import torch

from gomoku_env import GomokuEnv
from agents import HumanAgent
from aznet_cnn import PolicyValueNetCNN
from mcts import MCTSAgent
from play import play_game

BS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
WL = int(sys.argv[2]) if len(sys.argv) > 2 else 4
MODEL = sys.argv[3] if len(sys.argv) > 3 else f"az_cnn_{BS}x{BS}.pt"
N_SIM = int(sys.argv[4]) if len(sys.argv) > 4 else 400   # 对弈一步一搜，可比训练多搜些更强


def main():
    if not os.path.exists(MODEL):
        print(f"找不到模型 {MODEL}，请先训练：  python train_az_cnn.py")
        return

    # 1) 加载网络（结构必须和存盘时一致 → 同一个 PolicyValueNetCNN(BS)）
    net = PolicyValueNetCNN(BS)
    net.load_state_dict(torch.load(MODEL, map_location="cpu"))
    net.eval()

    # 2) 把网络包成棋手：MCTSAgent(net=...) —— 这就是 AI
    ai = MCTSAgent(BS, WL, n_simulations=N_SIM, c_puct=1.0,
                   name="AI", seed=0, net=net, device="cpu")
    human = HumanAgent()
    env = GomokuEnv(BS, WL)

    print("=" * 52)
    print(f" 跟你训练的 AlphaZero-CNN 下 {BS}×{BS}（{WL} 连胜），每步模拟 {N_SIM} 次")
    print(" 落子格式：行 列，例如  2 3 ；输 q 退出")
    print("=" * 52)

    side = input("你执黑(先手 X) 还是 白(后手 O)？ 输 b/w [默认 b]: ").strip().lower()

    # HumanAgent 和 MCTSAgent 都遵守同一接口，直接复用 play_game
    if side == "w":
        winner = play_game(env, ai, human, render=True)     # AI 执黑 / 你执白
        human_color = env.WHITE
    else:
        winner = play_game(env, human, ai, render=True)     # 你执黑 / AI 执白
        human_color = env.BLACK

    if winner is None:
        print("结果：平局！")
    elif winner == human_color:
        print("结果：你赢了 AlphaZero！🎉")
    else:
        print("结果：AlphaZero 赢了，再来一局？")


if __name__ == "__main__":
    main()
